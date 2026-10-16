"""Collect available Instagram stories once; no classification or messaging yet.

Python 3.10+. Run `python collector.py --login` once, then `python collector.py`.
"""

# 1. Imports and configuration: paths are relative to this project, not your shell.

# insta data collector - logs in, downloads all stories + grabs metadata, downloads image + data.
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from getpass import getpass
import json
import logging
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from instagrapi import Client
from instagrapi.exceptions import ClientError, TwoFactorRequired
from requests.exceptions import RequestException

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("collector")


def load_config():
    load_dotenv(ROOT / ".env")  # Existing shell variables take precedence.
    username = os.getenv("INSTAGRAM_USERNAME", "").strip()
    target = os.getenv("INSTAGRAM_TARGET", "zero2sudo").strip().lstrip("@")
    if not username:
        raise ValueError("Set INSTAGRAM_USERNAME in .env before running.")
    if not re.fullmatch(r"[A-Za-z0-9_.]{1,30}", target):
        raise ValueError("INSTAGRAM_TARGET must be an Instagram username.")
    data_dir = Path(os.getenv("COLLECTOR_DATA_DIR", "data")).expanduser()
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    return username, target, data_dir


# 2. Local persistence: atomic JSON writes and a lock prevent partial state/races.
def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def collector_lock(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "collector.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another collector is running for this data directory.") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# 3. Authentication: only --login performs a password login; normal runs reuse cookies.
def save_session(client, username, session_path):
    settings = client.get_settings()
    # The library doesn't persist client.username itself; keep our own identity tag.
    settings["collector_username"] = username
    write_json(session_path, settings)


def authenticate(client, username, session_path, login):
    if session_path.exists():
        if read_json(session_path).get("collector_username") != username:
            raise RuntimeError("Saved session belongs to another username. Use a new data directory.")
        # Explicit login migrates obsolete app version/hash fields while retaining
        # hardware identifiers and cookies. Normal collection keeps the saved profile.
        client.load_settings(session_path, override_app_version=login)
    # Apply after loading settings, so saved retry options cannot override these.
    client.set_retry_config(
        request_timeout=30, public_request_retries_count=1, session_retry_total=0,
        private_transport="curl",
    )
    if login:
        password = os.getenv("INSTAGRAM_PASSWORD") or getpass("Instagram password: ")
        try:
            try:
                logged_in = client.login(username, password)
            except TwoFactorRequired:
                code = getpass("Instagram two-factor code: ").strip()
                logged_in = client.login(username, password, verification_code=code)
            if not logged_in:
                raise RuntimeError("Instagram did not confirm login.")
        finally:
            # Preserve device identity even when the first login fails.
            save_session(client, username, session_path)
    else:
        if not session_path.exists():
            raise RuntimeError("No saved session. Run: python collector.py --login")
        if not client.user_id:
            raise RuntimeError("Saved session has no user ID. Run with --login to authenticate.")
    # Explicit credentials aren't needed for authenticated reads using saved settings.
    client.username = username


def error_diagnostics(client, exc):
    """Whitelist useful response fields; never dump requests, headers, or payloads."""
    payload = client.last_json if isinstance(client.last_json, dict) else {}
    response = getattr(exc, "response", None)
    if response is None:
        response = getattr(client, "last_response", None)
    details = {
        "exception": type(exc).__name__,
        "http_status": getattr(response, "status_code", None),
    }
    secrets = [os.getenv("INSTAGRAM_PASSWORD"), client.password, client.username]
    for values in (client.authorization_data, client.cookie_dict):
        secrets.extend(str(value) for value in values.values() if value is not None)

    def redact(value):
        for secret in sorted((s for s in secrets if isinstance(s, str) and s), key=len, reverse=True):
            value = value.replace(secret, "[redacted]")
        value = re.sub(r"https?://\S+", "[url redacted]", value)
        value = re.sub(r"[\w.+-]+@[\w.-]+", "[email redacted]", value)
        value = re.sub(r"\b\d{6,}\b", "[number redacted]", value)
        value = re.sub(r"\b[A-Za-z0-9_%+/=-]{40,}\b", "[token redacted]", value)
        return value[:500]

    for key in ("status", "error_type", "message"):
        value = payload.get(key)
        if isinstance(value, str):
            details[key] = redact(value)
    # Indicators are enough; nested challenge URLs and identifiers stay private.
    details["challenge_present"] = bool(payload.get("challenge") or payload.get("challenge_required"))
    details["two_factor_required"] = bool(payload.get("two_factor_required"))
    return details


# 4. Link extraction: preserve actual URLs supplied by Instagram, without guessing.
def extract_links(metadata):
    urls = [link.get("webUri") for link in metadata.get("links", [])]
    for sticker in metadata.get("stickers", []):
        link = sticker.get("story_link") or {}
        urls.append(link.get("url"))
    return list(dict.fromkeys(url for url in urls if url))


# 5. Story storage: save metadata first; mark complete only after downloading media.
def save_story(client, story, target, data_dir):
    story_id = str(story.pk)
    if not story_id.isdigit():
        raise ValueError("Instagram returned an invalid story ID.")
    folder = data_dir / "stories" / target / story_id
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder / "story.json"
    previous = read_json(record_path)
    # A completed record must also have its media; otherwise retry the download.
    if previous.get("complete"):
        filename = previous.get("media_file", "")
        media_path = folder / filename
        if filename and media_path.is_file() and media_path.stat().st_size > 0:
            return False

    metadata = story.model_dump(mode="json")
    record = {
        "story_id": story_id,
        "target": target,
        "story_url": f"https://www.instagram.com/stories/{target}/{story_id}/",
        "collected_at": previous.get("collected_at") or datetime.now(timezone.utc).isoformat(),
        "links": extract_links(metadata),
        "metadata": metadata,
        "complete": False,
    }
    write_json(record_path, record)
    url = story.video_url if story.media_type == 2 else story.thumbnail_url
    if not url:
        raise ValueError(f"Story {story_id} has no downloadable media URL.")
    # Download from the already-discovered CDN URL, avoiding another story-info call.
    downloaded = Path(client.story_download_by_url(str(url), filename="media", folder=folder))
    if not downloaded.is_file() or downloaded.stat().st_size == 0:
        raise RuntimeError(f"Story {story_id} did not produce a nonempty media file.")
    record.update(complete=True, media_file=downloaded.name)
    write_json(record_path, record)
    return True


# 6. Collection: cache the account ID and inspect all currently available stories.
def collect_once(client, target, data_dir):
    state_path = data_dir / "state.json"
    state = read_json(state_path)
    accounts = state.setdefault("accounts", {})
    if target not in accounts:
        accounts[target] = str(client.user_info_by_username_v1(target).pk)
        write_json(state_path, state)
    # Explicit authenticated endpoint: no anonymous/public fallback or extra retry path.
    stories = client.user_stories_v1(accounts[target])
    saved = 0
    failed = 0
    for story in stories:
        try:
            saved += save_story(client, story, target, data_dir)
        except ClientError:
            # Stop on Instagram errors (including throttling/authentication failures).
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            LOG.error("Story %s incomplete (%s); retry on the next run.", story.pk, type(exc).__name__)
            failed += 1
    state["last_fetch_at"] = datetime.now(timezone.utc).isoformat()
    state["last_fetch_target"] = target
    state["available_stories"] = len(stories)
    state["incomplete_stories"] = failed
    if not failed:
        state["last_success_at"] = state["last_fetch_at"]
    write_json(state_path, state)
    LOG.info("Available: %s; newly saved: %s; incomplete: %s", len(stories), saved, failed)
    return 1 if failed else 0


# 7. CLI: one run at a time; deliberately no automatic relogin/retry loop yet.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login", action="store_true", help="Explicitly authenticate and save a session, then collect")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Files include cookies and private media URLs. Restrict newly created files.
    os.umask(0o077)
    client = None
    try:
        username, target, data_dir = load_config()
        with collector_lock(data_dir):
            client = Client()
            session_path = data_dir / "session.json"
            authenticate(client, username, session_path, args.login)
            try:
                return collect_once(client, target, data_dir)
            finally:
                save_session(client, username, session_path)
    except ClientError as exc:
        LOG.error("Instagram stopped collection (%s). Inspect your account; do not retry repeatedly.", type(exc).__name__)
        if client is not None:
            LOG.error("Instagram diagnostics: %s", json.dumps(error_diagnostics(client, exc)))
        return 2
    except RequestException as exc:
        LOG.error("Network request failed (%s); collection is incomplete.", type(exc).__name__)
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        LOG.error("Collection stopped: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
