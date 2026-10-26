"""Send validated SWE internship results to Telegram, tracking each recipient."""
import argparse
from collections import Counter
from contextlib import closing, nullcontext
from datetime import datetime, timezone
from io import BytesIO
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import time

from dotenv import load_dotenv
import requests
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from collector import ROOT, collector_lock
from job_page import canonical_url

LOG = logging.getLogger("notifier")
register_heif_opener()


class TelegramControlError(RuntimeError):
    def __init__(self, code=None, retry_after=10):
        super().__init__("Telegram control request failed; check connectivity, token, and bot polling conflicts.")
        self.code = code
        self.retry_after = retry_after


# 1. Read only final, current decisions. Never use the model's raw label.
def candidates(data_dir, target, max_age_hours, now, bot=None):
    results = []
    review_decisions = {}
    if (data_dir / "telegram" / "reviews.sqlite3").exists():
        # Lazy import avoids a transport/review import cycle and keeps old setups working.
        from review import decisions, fingerprint
        bot = bot or os.getenv("TELEGRAM_BOT_TOKEN", "").split(":", 1)[0]
        review_decisions = decisions(data_dir, bot)
    for path in sorted((data_dir / "classifications" / target).glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("story_id") != path.stem or record.get("target") != target:
            raise ValueError(f"Classification identity mismatch: {path.name}")
        resolution = review_decisions.get((target, record.get("story_id"), fingerprint(record))) if review_decisions else None
        approved = resolution == "approved"
        label = record.get("decision", {}).get("classification")
        if (record.get("status") != "complete" or resolution == "skipped"
                or label not in (("swe_internship", "uncertain") if approved else ("swe_internship",))
                or (record.get("review_required") is not False and not approved)
                or record.get("job_status") in ("closed", "expired")):
            continue
        try:
            posted = datetime.fromisoformat(record["taken_at"].replace("Z", "+00:00"))
            age = now - posted.timestamp() if posted.tzinfo else -1
            links = list(dict.fromkeys(canonical_url(url) for url in record.get("links", [])))
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= age <= max_age_hours * 3600 or not links:
            continue
        decision = record["decision"]
        lines = ["SWE internship", str(decision.get("company") or "Company unspecified")[:200],
                 str(decision.get("role") or "Software engineering internship")[:300],
                 "", *links]
        story_url = record.get("story_url")
        if story_url:
            try:
                lines.extend(["", "Story: " + canonical_url(story_url)])
            except ValueError:
                pass
        text = "\n".join(lines)
        # Preserve complete links; skip instead of truncating an application URL.
        if len(text.encode("utf-16-le")) // 2 <= 4000:
            results.append((posted.timestamp(), record["story_id"], text))
    return sorted(results)


# 2. Durable, per-bot/per-chat/per-story state. Token secrets never enter SQLite.
class DeliveryStore:
    def __init__(self, path, readonly=False):
        if readonly and path.exists():
            self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        else:
            self.db = sqlite3.connect(":memory:" if readonly else path)
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS deliveries (
                    bot TEXT, chat TEXT, story TEXT, state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL,
                    next_attempt REAL NOT NULL DEFAULT 0, message_id INTEGER,
                    error TEXT, PRIMARY KEY (bot, chat, story)
                );
                CREATE TABLE IF NOT EXISTS cooldowns (bot TEXT PRIMARY KEY, until REAL NOT NULL);
            """)
        self.db.row_factory = sqlite3.Row

    def close(self):
        self.db.close()

    def get(self, bot, chat, story):
        return self.db.execute("SELECT * FROM deliveries WHERE bot=? AND chat=? AND story=?",
                               (bot, chat, story)).fetchone()

    def cooldown(self, bot):
        row = self.db.execute("SELECT until FROM cooldowns WHERE bot=?", (bot,)).fetchone()
        return row[0] if row else 0

    def begin(self, bot, chat, story, now):
        # Caller holds the process lock; commit before contacting Telegram.
        with self.db:
            self.db.execute("""INSERT INTO deliveries (bot,chat,story,state,attempts,updated)
                VALUES (?,?,?,'sending',1,?) ON CONFLICT(bot,chat,story) DO UPDATE SET
                state='sending', attempts=attempts+1, updated=excluded.updated,
                next_attempt=0, error=NULL""", (bot, chat, story, now))

    def finish(self, bot, chat, story, state, now, message_id=None, error=None, retry_after=0):
        with self.db:
            self.db.execute("""UPDATE deliveries SET state=?,updated=?,message_id=?,error=?,
                next_attempt=? WHERE bot=? AND chat=? AND story=?""",
                (state, now, message_id, error, now + retry_after, bot, chat, story))
            if retry_after:
                self.db.execute("INSERT INTO cooldowns VALUES (?,?) ON CONFLICT(bot) DO UPDATE SET until=excluded.until",
                                (bot, now + retry_after))

    def resolve(self, bot, chat, story, resolution):
        row = self.get(bot, chat, story)
        if not row or row["state"] not in ("sending", "unknown", "rejected", "retry"):
            raise ValueError("Select an existing unresolved delivery; sent deliveries cannot be reset.")
        self.finish(bot, chat, story, "sent" if resolution == "sent" else "retry", time.time())

    def rows(self, bot):
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM deliveries WHERE bot=? ORDER BY updated", (bot,))]


# 3. Telegram transport. Never log exception strings: request URLs contain tokens.
class TelegramClient:
    def __init__(self, token, session=None):
        if not re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", token):
            raise ValueError("Set a valid TELEGRAM_BOT_TOKEN from BotFather.")
        self.bot = token.split(":", 1)[0]
        self._base = "https://api.telegram.org/bot" + token + "/"
        self.session = session or requests.Session()

    def send(self, chat, text, reply_markup=None):
        payload = {"chat_id": chat, "text": text, "link_preview_options": {"is_disabled": True}}
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return self._send_message("sendMessage", payload)

    def send_review(self, chat, caption, markup, path):
        """Upload the saved media so reviewing doesn't require opening Instagram."""
        if path is None or path.stat().st_size > 45 * 1024 * 1024:
            return self.send(chat, caption + "\nMedia unavailable here; use the story link.", markup)
        if path.suffix.lower() in (".mp4", ".mov"):
            with path.open("rb") as handle:
                return self._send_message("sendVideo", {"chat_id": chat, "caption": caption,
                    "reply_markup": json.dumps(markup), "supports_streaming": "true"},
                    files={"video": ("story.mp4", handle, "video/mp4")})
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1280, 1280))
            buffer = BytesIO()
            image.save(buffer, "JPEG", quality=85)
        buffer.seek(0)
        return self._send_message("sendPhoto", {"chat_id": chat, "caption": caption,
            "reply_markup": json.dumps(markup)}, files={"photo": ("story.jpg", buffer, "image/jpeg")})

    def control(self, method, payload):
        """Poll/acknowledge callbacks without exposing token-bearing request URLs."""
        try:
            response = self.session.post(self._base + method, json=payload,
                                         timeout=(5, 30), allow_redirects=False)
            body = response.json()
        except (requests.RequestException, ValueError):
            raise TelegramControlError() from None
        if not isinstance(body, dict):
            raise TelegramControlError()
        if response.status_code != 200 or body.get("ok") is not True:
            params = body.get("parameters")
            delay = params.get("retry_after", 10) if isinstance(params, dict) else 10
            delay = delay if isinstance(delay, (int, float)) and math.isfinite(delay) else 10
            raise TelegramControlError(body.get("error_code", response.status_code), max(1, delay))
        return body.get("result")

    def _send_message(self, method, payload, files=None):
        try:
            kwargs = {"json": payload} if files is None else {"data": payload, "files": files}
            response = self.session.post(self._base + method, **kwargs,
                                         timeout=(5, 20), allow_redirects=False)
        except requests.ConnectTimeout:
            return {"state": "retry", "error": "connect_timeout", "retry_after": 60}
        except requests.RequestException:
            return {"state": "unknown", "error": "transport_outcome_unknown"}
        try:
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return {"state": "unknown", "error": "invalid_response"}
        if response.status_code == 200 and body.get("ok") is True:
            result = body.get("result")
            message_id = result.get("message_id") if isinstance(result, dict) else None
            if isinstance(message_id, int) and not isinstance(message_id, bool):
                return {"state": "sent", "message_id": message_id}
            return {"state": "unknown", "error": "missing_message_id"}
        code = body.get("error_code", response.status_code)
        if body.get("ok") is False and code == 429:
            parameters = body.get("parameters")
            delay = parameters.get("retry_after", 60) if isinstance(parameters, dict) else 60
            delay = delay if isinstance(delay, (int, float)) and math.isfinite(delay) else 60
            return {"state": "retry", "error": "rate_limited", "retry_after": max(1, delay)}
        if body.get("ok") is False and code in (400, 401, 403, 404, 409):
            return {"state": "rejected", "error": f"telegram_{code}"}
        return {"state": "unknown", "error": "server_outcome_unknown"}

    def list_chats(self):
        try:
            response = self.session.post(self._base + "getUpdates", json={"timeout": 0},
                                         timeout=(5, 20), allow_redirects=False)
            body = response.json()
            if (response.status_code != 200 or not isinstance(body, dict)
                    or body.get("ok") is not True or not isinstance(body.get("result"), list)):
                raise ValueError
            chats = {}
            for update in body["result"]:
                if not isinstance(update, dict):
                    continue
                for key in ("message", "channel_post", "my_chat_member"):
                    event = update.get(key)
                    chat = event.get("chat") if isinstance(event, dict) else None
                    if isinstance(chat, dict) and isinstance(chat.get("id"), int):
                        chats[str(chat["id"])] = chat.get("type", "unknown")
            return chats
        except (requests.RequestException, ValueError, KeyError, TypeError):
            raise RuntimeError("Cannot discover chats. Check token, network, and whether another bot gateway/webhook is active.") from None


# 4. Dispatch only unseen or explicitly retryable deliveries. No automatic resend
# after timeouts/crashes: Telegram sendMessage has no idempotency key.
def notify(items, chats, client, store, limit=20, dry_run=False, clock=time.time, sleep=time.sleep):
    counts = Counter()
    attempted = 0
    for _, story, text in items:
        for chat in chats:
            if attempted >= limit:
                return counts
            row = store.get(client.bot, chat, story)
            if row and (row["state"] != "retry" or row["next_attempt"] > clock()):
                counts["already_sent" if row["state"] == "sent" else row["state"]] += 1
                continue
            if store.cooldown(client.bot) > clock():
                counts["cooldown"] += 1
                return counts
            attempted += 1
            if dry_run:
                LOG.info("Would send story %s to chat %s:\n%s", story, chat, text)
                counts["pending"] += 1
                continue
            if attempted > 1:
                sleep(1.1)
            store.begin(client.bot, chat, story, clock())
            result = client.send(chat, text)
            store.finish(client.bot, chat, story, now=clock(), **result)
            counts[result["state"]] += 1
            LOG.info("Story %s, chat %s: %s", story, chat, result["state"])
            if result["state"] == "retry" or result.get("error") == "telegram_401":
                return counts
    return counts


# 5. Manual CLI; scheduler and Hermes can call this same command later.
def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview; no Telegram requests or disk writes")
    parser.add_argument("--chat-id", action="append", help="Recipient override; repeat for multiple chats")
    parser.add_argument("--limit", type=int, default=20, help="Maximum delivery attempts (default: 20)")
    parser.add_argument("--max-age-hours", type=float, default=24, help="Maximum story age (default: 24)")
    parser.add_argument("--list-chats", action="store_true", help="Discover IDs from recent bot updates; no sending")
    parser.add_argument("--status", action="store_true", help="Show delivery records; no Telegram requests")
    parser.add_argument("--resolve-story", help="Resolve an ambiguous/rejected delivery; requires one --chat-id")
    parser.add_argument("--resolution", choices=("sent", "retry"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.umask(0o077)
    try:
        if args.limit < 1 or not math.isfinite(args.max_age_hours) or args.max_age_hours <= 0:
            raise ValueError("Limit and maximum age must be positive.")
        actions = sum((args.list_chats, args.status, bool(args.resolve_story), args.dry_run))
        if actions > 1 or bool(args.resolve_story) != bool(args.resolution):
            raise ValueError("Select one action; --resolve-story and --resolution must be used together.")
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        client = TelegramClient(token or ("0:preview" if args.dry_run else ""))
        if args.list_chats:
            print(json.dumps(client.list_chats(), indent=2))
            return 0
        target = os.getenv("INSTAGRAM_TARGET", "zero2sudo").strip().lstrip("@")
        if not re.fullmatch(r"[A-Za-z0-9_.]{1,30}", target):
            raise ValueError("Invalid INSTAGRAM_TARGET.")
        data_dir = Path(os.getenv("COLLECTOR_DATA_DIR", "data")).expanduser()
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir
        directory = data_dir / "telegram"
        readonly = args.dry_run or args.status
        chats = args.chat_id if args.chat_id else os.getenv("TELEGRAM_CHAT_IDS", "").split(",")
        chats = list(dict.fromkeys(chat.strip() for chat in chats if chat.strip()))
        if not args.status and (not chats or any(not re.fullmatch(r"-?[0-9]+", chat) for chat in chats)):
            raise ValueError("Set numeric TELEGRAM_CHAT_IDS (comma-separated), or use --chat-id.")
        if args.resolve_story and (not args.chat_id or len(chats) != 1):
            raise ValueError("Resolution requires exactly one --chat-id.")
        # A separate lock serializes sender processes without blocking collection.
        with (nullcontext() if readonly else collector_lock(directory)):
            with closing(DeliveryStore(directory / "deliveries.sqlite3", readonly=readonly)) as store:
                if args.status:
                    print(json.dumps(store.rows(client.bot), indent=2))
                    return 0
                if args.resolve_story:
                    store.resolve(client.bot, chats[0], args.resolve_story, args.resolution)
                    LOG.info("Resolved story %s as %s", args.resolve_story, args.resolution)
                    return 0
                items = candidates(data_dir, target, args.max_age_hours, time.time())
                counts = notify(items, chats, client, store, args.limit, args.dry_run)
                LOG.info("Eligible stories: %s; deliveries: %s", len(items), dict(counts))
                if counts["unknown"] or counts["sending"] or counts["rejected"]:
                    LOG.warning("Use --status to review unresolved deliveries; they will not resend automatically.")
                return 1 if any(counts[s] for s in ("unknown", "sending", "rejected", "retry", "cooldown")) else 0
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
        LOG.error("Telegram sender stopped: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
