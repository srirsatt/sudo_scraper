"""Run collection, classification, and Telegram delivery once or every ten minutes."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from dotenv import load_dotenv

from collector import ROOT, read_json, write_json
from llama import ModelConfig
from notifier import TelegramClient
from subscriptions import recipient_chats

LOG = logging.getLogger("pipeline")


# 1. A pipeline lock is separate from the collector/classifier and sender locks.
@contextmanager
def pipeline_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "pipeline.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another pipeline is running for this data directory.") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# 2. All scripts use this Python environment and project directory, never a shell.
def build_steps(args):
    steps = []
    if not args.skip_collect:
        steps.append(("collect", [sys.executable, str(ROOT / "collector.py")]))
    command = [sys.executable, str(ROOT / "classifier.py"), "--mode", args.mode]
    for option, value in (("--provider", args.provider), ("--model", args.model),
                          ("--limit", args.classification_limit)):
        if value is not None:
            command.extend([option, str(value)])
    steps.append(("classify", command))
    if not args.skip_notify:
        steps.append(("notify", [sys.executable, str(ROOT / "notifier.py"),
                                 "--max-age-hours", str(args.max_age_hours),
                                 "--limit", str(args.notify_limit)]))
    return steps


def preflight(args, data_dir):
    """Validate local configuration before any collection/model/message requests."""
    target = os.getenv("INSTAGRAM_TARGET", "zero2sudo").strip().lstrip("@")
    import re
    if not re.fullmatch(r"[A-Za-z0-9_.]{1,30}", target):
        raise ValueError("Invalid INSTAGRAM_TARGET.")
    if not args.skip_collect:
        username = os.getenv("INSTAGRAM_USERNAME", "").strip()
        if not username:
            raise ValueError("Set INSTAGRAM_USERNAME before running collection.")
        session = read_json(data_dir / "session.json")
        if not session:
            raise ValueError("No saved Instagram session. Run python collector.py --login manually first.")
        if session.get("collector_username") != username:
            raise ValueError("Saved Instagram session belongs to a different account.")
    ModelConfig.from_env(model_override=args.model, provider_override=args.provider, mode_override=args.mode)
    if not args.skip_notify:
        TelegramClient(os.getenv("TELEGRAM_BOT_TOKEN", "").strip())
        recipient_chats()  # Validate; an empty audience can subscribe via /start later.


# 3. Persist stage transitions before/after execution. Child logs stay on the
# terminal; the local event log records timings and exit codes, not credentials.
def save_status(directory, record, event):
    write_json(directory / "last_run.json", record)
    with (directory / "events.jsonl").open("a") as handle:
        handle.write(json.dumps({"at": utc_now(), "run_id": record["run_id"],
                                 "event": event, "status": record["status"],
                                 "stages": record["stages"]}) + "\n")


def run_once(steps, directory, timeout=3600, runner=subprocess.run, clock=time.monotonic):
    started = clock()
    record = {"run_id": uuid4().hex, "started_at": utc_now(), "finished_at": None,
              "status": "running", "stages": []}
    save_status(directory, record, "started")
    for name, command in steps:
        stage_started = clock()
        stage = {"name": name, "started_at": utc_now(), "status": "running"}
        record["stages"].append(stage)
        save_status(directory, record, "stage_started")
        LOG.info("Starting %s", name)
        try:
            result = runner(command, cwd=ROOT, timeout=timeout, check=False)
            code = result.returncode
            stage.update(status="complete" if code == 0 else "failed", exit_code=code)
        except subprocess.TimeoutExpired:
            code = 124
            stage.update(status="timed_out", exit_code=code)
        except OSError:
            code = 1
            stage.update(status="failed", exit_code=code, error="process_start_failed")
        except KeyboardInterrupt:
            stage.update(status="interrupted", exit_code=130,
                         elapsed_seconds=round(clock() - stage_started, 2), finished_at=utc_now())
            record.update(status="interrupted", finished_at=utc_now(),
                          elapsed_seconds=round(clock() - started, 2))
            save_status(directory, record, "interrupted")
            raise
        stage.update(elapsed_seconds=round(clock() - stage_started, 2), finished_at=utc_now())
        # Collector exit 2 means an Instagram API/account error: stop watch mode
        # instead of automatically trying that account again ten minutes later.
        if name == "collect" and code == 2:
            record["status"] = "instagram_stopped"
        elif code:
            record["status"] = "failed"
        save_status(directory, record, "stage_finished")
        LOG.info("%s: %s (exit %s, %.1fs)", name, stage["status"], code, stage["elapsed_seconds"])
        if code:
            break
    if record["status"] == "running":
        record["status"] = "complete"
    record.update(finished_at=utc_now(), elapsed_seconds=round(clock() - started, 2))
    save_status(directory, record, "finished")
    LOG.info("Pipeline %s in %.1fs", record["status"], record["elapsed_seconds"])
    return record


# 4. Watch mode is sequential. A slow pass can delay the next check; it never
# launches overlapping passes. Ctrl-C stops it; existing stage caches survive.
def run_pipeline(steps, directory, watch=False, interval=600, timeout=3600,
                 once=run_once, clock=time.monotonic, sleep=time.sleep):
    while True:
        started = clock()
        record = once(steps, directory, timeout=timeout)
        if record["status"] == "instagram_stopped":
            LOG.error("Instagram stopped collection. Inspect your account before manually restarting the pipeline.")
            return 2
        if not watch:
            return 0 if record["status"] == "complete" else 1
        delay = max(0, interval - (clock() - started))
        LOG.info("Next pipeline pass in %.0fs; Ctrl-C to stop", delay)
        sleep(delay)


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Repeat until stopped; default interval: 600 seconds")
    parser.add_argument("--interval-seconds", type=float, default=600, help="Watch interval, minimum 60 seconds")
    parser.add_argument("--stage-timeout", type=float, default=3600, help="Maximum seconds per stage (default: 3600)")
    parser.add_argument("--mode", choices=("auto", "fast", "thorough"), default="auto")
    parser.add_argument("--provider", choices=("ollama", "fireworks", "openai", "custom"))
    parser.add_argument("--model")
    parser.add_argument("--classification-limit", type=int, help="Limit uncached stories per pass")
    parser.add_argument("--notify-limit", type=int, default=20, help="Limit recipient delivery attempts per pass")
    parser.add_argument("--max-age-hours", type=float, default=24, help="Maximum story age for Telegram delivery")
    parser.add_argument("--skip-collect", action="store_true", help="Use saved stories without contacting Instagram")
    parser.add_argument("--skip-notify", action="store_true", help="Collect/classify without sending Telegram messages")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only; no requests, subprocesses, or disk writes")
    parser.add_argument("--status", action="store_true", help="Read the last saved run; no requests or disk writes")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.umask(0o077)
    try:
        for value in (args.interval_seconds, args.stage_timeout, args.max_age_hours):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Intervals, timeouts, and maximum age must be finite and positive.")
        if args.interval_seconds < 60 or args.notify_limit < 1 or (args.classification_limit is not None and args.classification_limit < 1):
            raise ValueError("Watch interval must be at least 60 seconds; limits must be positive.")
        if sum((args.watch, args.dry_run, args.status)) > 1:
            raise ValueError("Choose one of --watch, --dry-run, or --status.")
        data_dir = Path(os.getenv("COLLECTOR_DATA_DIR", "data")).expanduser()
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir
        directory = data_dir / "pipeline"
        if args.status:
            print(json.dumps(read_json(directory / "last_run.json"), indent=2))
            return 0
        steps = build_steps(args)
        if args.dry_run:
            for name, command in steps:
                print(json.dumps({"stage": name, "command": command}))
            return 0
        preflight(args, data_dir)
        with pipeline_lock(directory):
            previous = read_json(directory / "last_run.json")
            if previous.get("status") == "running":
                LOG.warning("Previous pipeline was interrupted; resuming with saved stage caches and delivery tracking.")
            return run_pipeline(steps, directory, args.watch, args.interval_seconds, args.stage_timeout)
    except KeyboardInterrupt:
        LOG.info("Pipeline stopped by user.")
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        LOG.error("Pipeline stopped: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
