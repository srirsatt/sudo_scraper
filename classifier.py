"""Classify saved story images without making any Instagram requests."""

# 1. Load completed stories from disk. No Instagram login is needed here.
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from time import monotonic

from dotenv import load_dotenv

from collector import ROOT, collector_lock, read_json, write_json
from llama import Decision, ModelConfig, ModelOutputError, ModelUnavailable, PROMPT_VERSION, SYSTEM_PROMPT, PAGE_SYSTEM_PROMPT, VisionClassifier
from media import IMAGE_SUFFIXES, MEDIA_VERSION, VIDEO_SUFFIXES, MediaError
from job_page import JobPageClient, PAGE_VERSION

LOG = logging.getLogger("classifier")
POLICY_VERSION = "coding-eligibility-v2"
INTERNSHIP = re.compile(r"\b(?:intern(?:ship|ing)?s?|co[ -]?op)\b", re.I)
CODING = re.compile(
    r"\b(?:SWE|SDE|software|coding|code|programming|programmers?|developers?|scripting|automation|"
    r"Python|JavaScript|TypeScript|Java|SQL|Rust|Golang|PyTorch|TensorFlow|JAX)\b|C\+\+|C#|"
    r"\b(?:AI|ML|AI/ML|machine learning|artificial intelligence|computer vision|"
    r"data scien(?:ce|tist)|data engineer(?:ing)?|robotics|autonomy)\b|"
    r"\b(?:front[ -]?end|back[ -]?end|full[ -]?stack|mobile|iOS|Android|web)\s+(?:engineer(?:ing)?|develop(?:er|ment))\b",
    re.I,
)
NON_SOFTWARE_ROLE = re.compile(
    r"^(?:technical\s+)?(?:product|program|project)\s+(?:manager|management)\b|\bPM\s+MBA\b",
    re.I,
)
NEW_GRAD = re.compile(r"\b(?:new[ -]?grad(?:uate)?s?|new college grad(?:uate)?)\b", re.I)
NOT_OPEN = re.compile(r"\b(?:no formal applications|no applications (?:are )?open|"
                      r"exploring whether to open|gauging interest)\b", re.I)


def apply_eligibility_policy(decision):
    """Accept coding work in any internship title/department; retain opening checks."""
    if decision.classification != "swe_internship":
        return decision, []
    evidence = " ".join(decision.evidence)
    role = decision.role or ""
    combined = role + " " + evidence
    coding = bool(CODING.search(combined))
    if NEW_GRAD.search(role) and not INTERNSHIP.search(role):
        note = "The advertised role is a new-grad job, not an internship/co-op."
        return decision.model_copy(update={"classification": "irrelevant", "reason": note}), [note]
    if NON_SOFTWARE_ROLE.search(role) and not coding:
        note = "The management role does not establish coding/development work."
        return decision.model_copy(update={"classification": "irrelevant", "reason": note}), [note]
    notes = []
    if not INTERNSHIP.search(combined):
        notes.append("Role and evidence do not establish an internship or co-op.")
    if not coding:
        notes.append("Role and evidence do not establish coding/development work.")
    if NOT_OPEN.search(evidence):
        notes.append("The role is only being considered; an actual opening is not established.")
    if notes:
        return decision.model_copy(update={"classification": "uncertain", "reason": " ".join(notes)}), notes
    return decision, []


def load_stories(data_dir, target):
    stories = []
    for path in (data_dir / "stories" / target).glob("*/story.json"):
        record = read_json(path)
        if not record.get("complete"):
            continue
        filename = record.get("media_file", "")
        if not filename or Path(filename).name != filename:
            raise ValueError(f"Invalid media filename in {path}.")
        if record.get("target") != target or str(record.get("story_id")) != path.parent.name:
            raise ValueError(f"Story identity does not match its folder: {path}.")
        record["media_path"] = path.parent / filename
        if not record["media_path"].is_file() or record["media_path"].stat().st_size == 0:
            raise ValueError(f"Missing/empty media for story {record['story_id']}; rerun the collector.")
        stories.append(record)
    return sorted(stories, key=lambda s: (s["metadata"].get("taken_at", ""), s["story_id"]))


# 2. Context: at most one neighboring story on either side, within 30 minutes.
def nearby_images(stories, index):
    current = stories[index]
    context = []
    for neighbor in (index - 1, index + 1):
        if not 0 <= neighbor < len(stories):
            continue
        other = stories[neighbor]
        if other["media_path"].suffix.lower() not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            continue
        try:
            a = datetime.fromisoformat(current["metadata"]["taken_at"])
            b = datetime.fromisoformat(other["metadata"]["taken_at"])
            close = abs((a - b).total_seconds()) <= 1800
        except (KeyError, TypeError, ValueError):
            close = False
        if close:
            context.append(other)
    return context


# 3. Cache keys cover the model, prompt, images, metadata, links, and context.
def input_fingerprint(story, context, config, page=None):
    digest = hashlib.sha256()
    digest.update(json.dumps([PROMPT_VERSION, SYSTEM_PROMPT, MEDIA_VERSION, POLICY_VERSION, config.base_url, config.model, config.api_style, config.context_length, config.provider, config.max_output_tokens, config.mode, config.ollama_think if config.api_style == "ollama" else None]).encode())
    if config.mode == "auto":
        digest.update(json.dumps([PAGE_VERSION, PAGE_SYSTEM_PROMPT, page.get("content_hash") if page else None]).encode())
    for item in [story, *context]:
        digest.update(json.dumps({k: v for k, v in item.items() if k != "media_path"}, sort_keys=True).encode())
        with item["media_path"].open("rb") as media:
            for chunk in iter(lambda: media.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


# 4. Save decisions separately from collected evidence; errors remain retryable.
def classify_saved(data_dir, target, model, limit=None, force=False, dry_run=False, story_ids=None, page_client=None):
    stories = load_stories(data_dir, target)
    selected = set(story_ids) if story_ids else None
    if selected:
        missing = selected - {story["story_id"] for story in stories}
        if missing:
            raise ValueError("No completed saved stories for IDs: " + ", ".join(sorted(missing)))
    if not stories:
        LOG.warning("No completed stories found for %s. Check COLLECTOR_DATA_DIR and INSTAGRAM_TARGET.", target)
    output_dir = data_dir / "classifications" / target
    counts = Counter()
    processed = 0
    if model.config.mode == "auto" and page_client is None:
        page_client = JobPageClient(data_dir / "job_pages")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    for index, story in enumerate(stories):
        if selected and story["story_id"] not in selected:
            continue
        if limit is not None and processed >= limit:
            break
        context = [] if model.config.mode in ("auto", "fast") else nearby_images(stories, index)
        page = None
        if model.config.mode == "auto":
            links = story.get("links", [])
            page = page_client.lookup(links[0], cached_only=dry_run) if len(links) == 1 else {"status": "unknown", "reason": "no_single_job_link"}
        fingerprint = input_fingerprint(story, context, model.config, page)
        output_path = output_dir / (story["story_id"] + ".json")
        previous = read_json(output_path)
        if not force and previous.get("status") == "complete" and previous.get("input_fingerprint") == fingerprint:
            counts["cached"] += 1
            continue
        processed += 1
        if dry_run:
            LOG.info("Would classify story %s with %s context images", story["story_id"], len(context))
            counts["pending"] += 1
            continue
        record = {
            "story_id": story["story_id"], "target": target,
            "story_url": story.get("story_url"), "links": story.get("links", []),
            "taken_at": story["metadata"].get("taken_at"),
            "context_story_ids": [item["story_id"] for item in context],
            "model": model.config.model, "prompt_version": PROMPT_VERSION,
            "provider": model.config.provider,
            "mode": model.config.mode,
            "input_fingerprint": fingerprint,
            "classified_at": datetime.now(timezone.utc).isoformat(),
        }
        LOG.info("Classifying story %s with %s context stories using %s", story["story_id"], len(context), model.config.model)
        started = monotonic()
        try:
            source = "vision"
            if page is not None:
                record["job_page"] = {k: v for k, v in page.items() if k != "description"}
            if page and page.get("status") == "usable":
                LOG.info("Using job text from %s", page.get("source"))
                try:
                    decision = model.classify_text(page)
                    record["page_decision"] = decision.model_dump()
                    if isinstance(getattr(model, "last_metrics", None), dict):
                        record["page_metrics"] = model.last_metrics
                    if decision.classification != "uncertain":
                        source = "job_page"
                    else:
                        record["fallback_reason"] = "job_text_uncertain"
                except ModelOutputError:
                    record["fallback_reason"] = "invalid_job_text_response"
            elif page is not None:
                record["fallback_reason"] = page.get("reason", "job_text_unavailable")
            if source == "vision":
                if model.config.mode == "auto":
                    LOG.info("Falling back to one-frame vision: %s", record["fallback_reason"])
                decision = model.classify(story, context)
        except ModelUnavailable as exc:
            record.update(status="error", error=str(exc), error_type=exc.kind)
            record["inference_seconds"] = round(monotonic() - started, 2)
            write_json(output_path, record)
            raise
        except (ModelOutputError, MediaError, OSError) as exc:
            message = str(exc) if isinstance(exc, (ModelOutputError, MediaError)) else type(exc).__name__
            record.update(status="error", error=message)
            record["inference_seconds"] = round(monotonic() - started, 2)
            if isinstance(getattr(model, "last_metrics", None), dict):
                record["model_metrics"] = model.last_metrics
            if isinstance(exc, ModelOutputError) and exc.content:
                record["model_output"] = exc.content
            write_json(output_path, record)
            counts["errors"] += 1
            LOG.error("Story %s: %s; result remains retryable", story["story_id"], message)
            continue
        record["model_decision"] = decision.model_dump()
        record["classification_source"] = source
        if isinstance(getattr(model, "last_metrics", None), dict):
            record["model_metrics"] = model.last_metrics
        record["inference_seconds"] = round(monotonic() - started, 2)
        decision, notes = apply_eligibility_policy(decision)
        record["policy_version"] = POLICY_VERSION
        record["policy_notes"] = notes
        is_video = story["media_path"].suffix.lower() in VIDEO_SUFFIXES
        record["media_coverage"] = "job_page_text" if source == "job_page" else (("single_frame_no_audio" if model.config.mode in ("auto", "fast") else "sampled_frames_no_audio") if is_video else "image")
        # Sampled frames cannot establish that an entire video's content is irrelevant.
        if source == "vision" and is_video and decision.classification == "irrelevant":
            decision = decision.model_copy(update={
                "classification": "uncertain",
                "reason": "Visible frames were classified as irrelevant. Unsampled frames and audio remain unreviewed.",
            })
            record["policy_notes"].append("Video negatives are held for review because coverage is incomplete.")
        record["status"] = "complete"
        record["decision"] = decision.model_dump()
        record["review_required"] = decision.classification == "uncertain" or (source == "vision" and is_video)
        if page and page.get("availability") in ("expired", "closed"):
            record["review_required"] = True
            record["job_status"] = page["availability"]
            record["policy_notes"].append("The linked job page is closed or has a past deadline; don't treat this as a current opening.")
        if decision.classification == "swe_internship" and not record["links"]:
            record["review_required"] = True
        write_json(output_path, record)
        counts[decision.classification] += 1
        LOG.info("Story %s: %s (%.1fs)", story["story_id"], decision.classification, record["inference_seconds"])
    LOG.info("Results: %s", dict(counts))
    return counts


# 5. CLI: run manually on a few stories before enabling the full batch.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Maximum uncached stories to process")
    parser.add_argument("--force", action="store_true", help="Reclassify cached stories")
    parser.add_argument("--model", help="Override LLM_MODEL for this run")
    parser.add_argument("--provider", choices=("ollama", "fireworks", "openai", "custom"), help="Override LLM_PROVIDER for this run")
    parser.add_argument("--mode", choices=("auto", "fast", "thorough"), help="Auto: job-page text first, vision fallback; fast/thorough: vision only")
    parser.add_argument("--story-id", action="append", help="Classify only this saved story ID; repeat to select multiple stories")
    parser.add_argument("--dry-run", action="store_true", help="List pending inputs without calling a model")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.umask(0o077)
    load_dotenv(ROOT / ".env")
    try:
        target = os.getenv("INSTAGRAM_TARGET", "zero2sudo").strip().lstrip("@")
        if not target or Path(target).name != target or target in (".", ".."):
            raise ValueError("INSTAGRAM_TARGET must be a username, not a path.")
        data_dir = Path(os.getenv("COLLECTOR_DATA_DIR", "data")).expanduser()
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir
        config = ModelConfig.from_env(model_override=args.model, provider_override=args.provider, mode_override=args.mode)
        model = VisionClassifier(config)
        with collector_lock(data_dir):
            counts = classify_saved(data_dir, target, model, args.limit, args.force, args.dry_run, args.story_id)
        return 1 if counts["errors"] else 0
    except (OSError, ValueError, RuntimeError) as exc:
        LOG.error("Classification stopped: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
