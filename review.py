"""Review held stories in a private Telegram chat with Approve / Skip buttons."""
import argparse
from contextlib import closing
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv

from collector import ROOT, collector_lock, read_json
from job_page import canonical_url
from notifier import DeliveryStore, TelegramClient, TelegramControlError, candidates, notify

LOG = logging.getLogger("review")


def fingerprint(record):
    """Bind a human decision to these exact classifier inputs and output label."""
    payload = {key: record.get(key) for key in ("input_fingerprint", "decision", "links", "job_status", "review_required")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_records(data_dir, target, max_age, now):
    records = []
    for path in sorted((data_dir / "classifications" / target).glob("*.json")):
        record = read_json(path)
        if record.get("story_id") != path.stem or record.get("target") != target:
            raise ValueError("Classification identity mismatch: " + path.name)
        if (record.get("status") != "complete" or not record.get("input_fingerprint")
                or record.get("decision", {}).get("classification") not in ("swe_internship", "uncertain")
                or record.get("review_required") is not True
                or record.get("job_status") in ("closed", "expired")):
            continue
        try:
            stamp = datetime.fromisoformat(record["taken_at"].replace("Z", "+00:00"))
            age = now - stamp.timestamp() if stamp.tzinfo else -1
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= age <= max_age * 3600:
            records.append(record)
    return sorted(records, key=lambda record: (record["taken_at"], record["story_id"]))


class ReviewStore:
    def __init__(self, path, readonly=False):
        if readonly and path.exists():
            self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        else:
            self.db = sqlite3.connect(":memory:" if readonly else path)
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id TEXT PRIMARY KEY, bot TEXT, target TEXT, story TEXT, fingerprint TEXT,
                    delivery_state TEXT NOT NULL DEFAULT 'pending', message_id INTEGER,
                    retry_at REAL NOT NULL DEFAULT 0, error TEXT, decision TEXT,
                    reviewer TEXT, decided_at REAL,
                    UNIQUE(bot,target,story,fingerprint)
                );
                CREATE TABLE IF NOT EXISTS bot_state (
                    bot TEXT PRIMARY KEY, next_update INTEGER NOT NULL DEFAULT 0,
                    cooldown REAL NOT NULL DEFAULT 0
                );
            """)
        self.db.row_factory = sqlite3.Row

    def close(self):
        self.db.close()

    def state(self, bot):
        row = self.db.execute("SELECT * FROM bot_state WHERE bot=?", (bot,)).fetchone()
        return dict(row) if row else {"next_update": 0, "cooldown": 0}

    def advance(self, bot, offset):
        with self.db:
            self.db.execute("INSERT INTO bot_state(bot,next_update) VALUES (?,?) ON CONFLICT(bot) DO UPDATE SET next_update=MAX(next_update,excluded.next_update)", (bot, offset))

    def get(self, card_id):
        row = self.db.execute("SELECT * FROM reviews WHERE id=?", (card_id,)).fetchone()
        return dict(row) if row else None

    def find(self, bot, record):
        row = self.db.execute("SELECT * FROM reviews WHERE bot=? AND target=? AND story=? AND fingerprint=?",
                              (bot, record["target"], record["story_id"], fingerprint(record))).fetchone()
        return dict(row) if row else None

    def enqueue(self, bot, record):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO reviews(id,bot,target,story,fingerprint) VALUES (?,?,?,?,?)",
                            (uuid4().hex, bot, record["target"], record["story_id"], fingerprint(record)))
        return self.find(bot, record)

    def delivery(self, card_id, state, now, message_id=None, error=None, retry_after=0):
        with self.db:
            self.db.execute("UPDATE reviews SET delivery_state=?,message_id=?,error=?,retry_at=? WHERE id=?",
                            (state, message_id, error, now + retry_after, card_id))
            if retry_after:
                bot = self.get(card_id)["bot"]
                self.db.execute("INSERT INTO bot_state(bot,cooldown) VALUES (?,?) ON CONFLICT(bot) DO UPDATE SET cooldown=excluded.cooldown", (bot, now + retry_after))

    def decide(self, card_id, decision, user, message_id):
        with self.db:
            self.db.execute("UPDATE reviews SET decision=?,reviewer=?,decided_at=?,message_id=?,delivery_state='sent' WHERE id=? AND decision IS NULL",
                            (decision, user, time.time(), message_id, card_id))

    def retry(self, bot, card_id):
        row = self.get(card_id)
        if not row or row["bot"] != bot or row["decision"] or row["delivery_state"] not in ("unknown", "sending", "rejected"):
            raise ValueError("Only an unresolved ambiguous/rejected card can be retried.")
        self.delivery(card_id, "pending", time.time())

    def rows(self, bot):
        return [dict(row) for row in self.db.execute("SELECT * FROM reviews WHERE bot=? ORDER BY rowid", (bot,))]


def decisions(data_dir, bot):
    path = data_dir / "telegram" / "reviews.sqlite3"
    if not path.exists():
        return {}
    with closing(ReviewStore(path, readonly=True)) as store:
        return {(r["target"], r["story"], r["fingerprint"]): r["decision"]
                for r in store.rows(bot) if r["decision"]}


def clip(text, units=900):
    return text.encode("utf-16-le")[:units * 2].decode("utf-16-le", errors="ignore")


def markup(record, card_id, buttons=True):
    rows = []
    if buttons:
        rows.append([{"text": "Approve", "callback_data": f"r:{card_id}:a"},
                     {"text": "Skip", "callback_data": f"r:{card_id}:s"}])
    for index, url in enumerate(record.get("links", [])[:3]):
        try:
            rows.append([{"text": "Application" if index == 0 else "Another link", "url": canonical_url(url)}])
        except ValueError:
            pass
    if record.get("story_url"):
        try:
            rows.append([{"text": "Original story", "url": canonical_url(record["story_url"])}])
        except ValueError:
            pass
    return {"inline_keyboard": rows}


def caption(record):
    decision = record["decision"]
    parts = ["Review internship", str(decision.get("company") or "Company unspecified"),
             str(decision.get("role") or "Role unclear"), "", str(decision.get("reason") or ""),
             "", "Evidence: " + " | ".join(decision.get("evidence", []))]
    if not record.get("links"):
        parts.insert(3, "No application link: approval alone cannot send this.")
    return clip("\n".join(parts))


def send_cards(records, data_dir, target, chat, client, store, limit=5, dry_run=False, sleep=time.sleep):
    sent = 0
    for record in records:
        row = store.find(client.bot, record)
        if row and (row["decision"] or row["delivery_state"] not in ("pending", "retry") or row["retry_at"] > time.time()):
            continue
        if store.state(client.bot)["cooldown"] > time.time() or sent >= limit:
            break
        if dry_run:
            LOG.info("Would review %s: %s", record["story_id"], caption(record))
            sent += 1
            continue
        row = row or store.enqueue(client.bot, record)
        story_dir = data_dir / "stories" / target / record["story_id"]
        story = read_json(story_dir / "story.json")
        filename = story.get("media_file", "")
        path = story_dir / filename if filename and Path(filename).name == filename else None
        if path is not None and not path.is_file():
            path = None
        if sent:
            sleep(1.1)
        store.delivery(row["id"], "sending", time.time())
        try:
            result = client.send_review(chat, caption(record), markup(record, row["id"]), path)
        except (OSError, ValueError):
            result = {"state": "rejected", "error": "media_preparation_failed"}
        store.delivery(row["id"], now=time.time(), **result)
        sent += 1
        LOG.info("Review story %s: %s", record["story_id"], result["state"])
        if result["state"] == "retry" or result.get("error") == "telegram_401":
            break
    return sent


def handle_callback(query, data_dir, target, chat, user, client, store):
    """Only the registered reviewer, chat, card, message, and current version count."""
    message = query.get("message") or {}
    sender = query.get("from") or {}
    origin = message.get("chat") or {}
    if (str(sender.get("id")) != user or str(origin.get("id")) != chat
            or origin.get("type") != "private"):
        return "This review belongs to the configured reviewer."
    match = re.fullmatch(r"r:([a-f0-9]{32}):([as])", str(query.get("data", "")))
    row = store.get(match[1]) if match else None
    if (not row or row["bot"] != client.bot or row["target"] != target
            or row["delivery_state"] not in ("sent", "unknown", "sending")
            or not isinstance(message.get("message_id"), int)
            or (row["message_id"] is not None and row["message_id"] != message["message_id"])):
        return "Unknown review card."
    if row["decision"]:
        return "Already " + row["decision"] + "."
    record = read_json(data_dir / "classifications" / target / (row["story"] + ".json"))
    if (not record or record.get("status") != "complete" or fingerprint(record) != row["fingerprint"]):
        return "This story changed; use its new review card."
    decision = "approved" if match[2] == "a" else "skipped"
    store.decide(row["id"], decision, user, message["message_id"])
    try:
        media_message = "caption" in message or "photo" in message or "video" in message
        client.control("editMessageCaption" if media_message else "editMessageText", {
            "chat_id": chat, "message_id": message["message_id"],
            "caption" if media_message else "text": clip(decision.title() + "\n\n" + caption(record)),
            "reply_markup": markup(record, row["id"], buttons=False)})
    except TelegramControlError:
        LOG.warning("Decision saved; couldn't remove the old review buttons.")
    return "Approved. Delivery still checks age, links, and job status." if decision == "approved" else "Skipped."


def poll(data_dir, target, chat, user, client, store, timeout=0):
    offset = store.state(client.bot)["next_update"]
    updates = client.control("getUpdates", {"offset": offset, "timeout": timeout,
        "allowed_updates": ["callback_query", "message", "channel_post", "my_chat_member"]})
    if not isinstance(updates, list):
        raise TelegramControlError()
    for update in sorted(updates, key=lambda item: item.get("update_id", -1) if isinstance(item, dict) else -1):
        if not isinstance(update, dict) or not isinstance(update.get("update_id"), int):
            continue
        if update["update_id"] < offset:
            continue
        query = update.get("callback_query")
        if isinstance(query, dict):
            text = handle_callback(query, data_dir, target, chat, user, client, store)
            try:
                client.control("answerCallbackQuery", {"callback_query_id": query["id"], "text": text})
            except (TelegramControlError, KeyError):
                LOG.warning("Review persisted; couldn't acknowledge a callback.")
        store.advance(client.bot, update["update_id"] + 1)


def deliver_approved(data_dir, target, max_age, client, store):
    approved = {row["story"] for row in store.rows(client.bot) if row["target"] == target and row["decision"] == "approved"}
    chats = list(dict.fromkeys(chat.strip() for chat in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if chat.strip()))
    if not approved or not chats:
        return
    if any(not re.fullmatch(r"-?[0-9]+", chat) for chat in chats):
        raise ValueError("TELEGRAM_CHAT_IDS must contain numeric recipient IDs.")
    try:
        with collector_lock(data_dir / "telegram"):
            with closing(DeliveryStore(data_dir / "telegram" / "deliveries.sqlite3")) as deliveries:
                items = [item for item in candidates(data_dir, target, max_age, time.time(), bot=client.bot) if item[1] in approved]
                notify(items, chats, client, deliveries)
    except RuntimeError:
        LOG.info("Sender busy; approved deliveries will retry on the next pass.")


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Keep polling buttons and queuing new review cards")
    parser.add_argument("--dry-run", action="store_true", help="Preview without network requests or disk writes")
    parser.add_argument("--status", action="store_true", help="Show saved review state")
    parser.add_argument("--retry-card", help="Retry an ambiguous/rejected card after checking your chat")
    parser.add_argument("--max-age-hours", type=float, default=24)
    parser.add_argument("--limit", type=int, default=5, help="Maximum review-card attempts per pass")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.umask(0o077)
    try:
        if sum((args.watch, args.dry_run, args.status, bool(args.retry_card))) > 1:
            raise ValueError("Choose one action.")
        if not math.isfinite(args.max_age_hours) or args.max_age_hours <= 0 or args.limit < 1:
            raise ValueError("Maximum age and limit must be positive.")
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        client = TelegramClient(token or ("0:preview" if args.dry_run else ""))
        target = os.getenv("INSTAGRAM_TARGET", "zero2sudo").strip().lstrip("@")
        if not re.fullmatch(r"[A-Za-z0-9_.]{1,30}", target):
            raise ValueError("Invalid INSTAGRAM_TARGET.")
        data_dir = Path(os.getenv("COLLECTOR_DATA_DIR", "data")).expanduser()
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir
        path = data_dir / "telegram" / "reviews.sqlite3"
        if args.status or args.dry_run:
            with closing(ReviewStore(path, readonly=True)) as store:
                if args.status:
                    print(json.dumps(store.rows(client.bot), indent=2))
                else:
                    records = load_records(data_dir, target, args.max_age_hours, time.time())
                    send_cards(records, data_dir, target, "preview", client, store, args.limit, dry_run=True)
            return 0
        chat = os.getenv("TELEGRAM_REVIEW_CHAT_ID", "").strip()
        user = os.getenv("TELEGRAM_REVIEW_USER_ID", "").strip() or chat
        if not re.fullmatch(r"[1-9][0-9]*", chat) or not re.fullmatch(r"[1-9][0-9]*", user):
            raise ValueError("Set TELEGRAM_REVIEW_CHAT_ID to your private chat ID; reviewer user ID defaults to it.")
        with collector_lock(data_dir / "telegram" / "review_worker"):
            with closing(ReviewStore(path)) as store:
                if args.retry_card:
                    store.retry(client.bot, args.retry_card)
                    return 0
                while True:
                    try:
                        poll(data_dir, target, chat, user, client, store)
                        records = load_records(data_dir, target, args.max_age_hours, time.time())
                        send_cards(records, data_dir, target, chat, client, store, args.limit)
                        deliver_approved(data_dir, target, args.max_age_hours, client, store)
                        if not args.watch:
                            return 0
                        poll(data_dir, target, chat, user, client, store, timeout=20)
                    except TelegramControlError as exc:
                        if not args.watch or exc.code in (401, 403, 409):
                            raise
                        LOG.warning("Telegram polling unavailable; retrying after %.0fs.", exc.retry_after)
                        time.sleep(exc.retry_after)
    except KeyboardInterrupt:
        LOG.info("Review worker stopped.")
        return 130
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        LOG.error("Review stopped: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
