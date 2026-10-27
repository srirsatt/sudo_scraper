"""Private Telegram subscriptions persisted in the local .env recipient list."""
import fcntl
import os
from pathlib import Path
import re

from dotenv import dotenv_values, set_key

from collector import ROOT


def recipient_chats(env_path=None):
    """Read fresh recipients, including changes made after a watcher started.

    Only this setting prefers .env over the inherited process environment.
    An explicitly empty file setting must not revive unsubscribed recipients.
    """
    path = Path(env_path) if env_path is not None else ROOT / ".env"
    values = dotenv_values(path, interpolate=False) if path.exists() else {}
    raw = values.get("TELEGRAM_CHAT_IDS", os.getenv("TELEGRAM_CHAT_IDS", "")) or ""
    chats = list(dict.fromkeys(chat.strip() for chat in raw.split(",") if chat.strip()))
    if any(not re.fullmatch(r"-?[0-9]+", chat) for chat in chats):
        raise ValueError("TELEGRAM_CHAT_IDS must contain numeric recipient IDs.")
    return chats


def set_subscription(chat, subscribed, env_path=None):
    """Serialize read/modify/write and atomically preserve other .env settings."""
    if not re.fullmatch(r"[1-9][0-9]*", chat):
        raise ValueError("Subscriptions require a positive private chat ID.")
    path = Path(env_path) if env_path is not None else ROOT / ".env"
    with path.with_name(path.name + ".lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        chats = recipient_chats(path)
        if subscribed == (chat in chats):
            return False
        if subscribed:
            chats.append(chat)
        else:
            chats.remove(chat)
        if path.exists():
            path.chmod(0o600)
        set_key(path, "TELEGRAM_CHAT_IDS", ",".join(chats), quote_mode="always")
        path.chmod(0o600)
    return True


def handle_subscription(message, env_path=None):
    """Only a person's own private /start or /stop changes their subscription."""
    if not isinstance(message, dict):
        return None
    sender, chat = message.get("from"), message.get("chat")
    if not isinstance(sender, dict) or not isinstance(chat, dict):
        return None
    chat_id = chat.get("id")
    if (chat.get("type") != "private" or type(chat_id) is not int or chat_id <= 0
            or type(sender.get("id")) is not int or sender["id"] != chat_id
            or sender.get("is_bot") is not False
            or any(key in message for key in ("forward_origin", "forward_date", "sender_chat"))):
        return None
    text = message.get("text")
    if not isinstance(text, str):
        return None
    # Deep links deliver /start plus an optional payload, never a recipient ID.
    command = re.fullmatch(r"/(start|stop)(?:\s+[A-Za-z0-9_-]{1,64})?", text.strip())
    if not command:
        return None
    subscribed = command[1] == "start"
    set_subscription(str(chat_id), subscribed, env_path)
    reply = ("You're subscribed to coding internship alerts! Eligible recent postings "
             "will arrive on upcoming checks. Send /stop to unsubscribe." if subscribed else
             "You've unsubscribed from internship alerts. An alert already being sent may still "
             "arrive. Send /start to subscribe again.")
    return str(chat_id), reply
