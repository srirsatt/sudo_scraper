"""Offline signup, unsubscribe, live recipient loading, and polling tests."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from dotenv import dotenv_values

from notifier import main as notify_main
from review import ReviewStore, deliver_approved, poll
from subscriptions import handle_subscription, recipient_chats, set_subscription


class SubscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = self.root / ".env"
        self.env.write_text("# Keep credentials and reviewer settings\nTOKEN='test-secret'\n"
                            "TELEGRAM_CHAT_IDS='11,-22'\nTELEGRAM_REVIEW_CHAT_ID=11\n")
        patcher = patch("subscriptions.ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def message(self, text="/start internships", chat=33):
        return {"text": text, "chat": {"id": chat, "type": "private"},
                "from": {"id": chat, "is_bot": False}}

    def test_deep_link_signup_preserves_settings_and_repeated_start_is_idempotent(self):
        reply = handle_subscription(self.message())
        self.assertEqual(reply[0], "33")
        self.assertIn("subscribed", reply[1])
        handle_subscription(self.message("/start"))
        self.assertEqual(recipient_chats(), ["11", "-22", "33"])
        values = dotenv_values(self.env)
        self.assertEqual(values["TOKEN"], "test-secret")
        self.assertEqual(values["TELEGRAM_REVIEW_CHAT_ID"], "11")
        self.assertIn("# Keep credentials", self.env.read_text())
        self.assertEqual(stat.S_IMODE(self.env.stat().st_mode), 0o600)

    def test_stop_and_restart_reload_despite_stale_inherited_environment(self):
        with patch.dict(os.environ, {"TELEGRAM_CHAT_IDS": "11,-22"}):
            handle_subscription(self.message())
            self.assertIn("33", recipient_chats())
            handle_subscription(self.message("/stop", 11))
            self.assertEqual(recipient_chats(), ["-22", "33"])
            handle_subscription(self.message("/stop", 33))
            self.assertEqual(recipient_chats(), ["-22"])
            handle_subscription(self.message("/start", 11))
            self.assertEqual(recipient_chats(), ["-22", "11"])

    def test_empty_file_value_does_not_resurrect_last_unsubscriber(self):
        self.env.write_text("TELEGRAM_CHAT_IDS=11\n")
        with patch.dict(os.environ, {"TELEGRAM_CHAT_IDS": "11"}):
            handle_subscription(self.message("/stop", 11))
            self.assertEqual(recipient_chats(), [])
            with patch("notifier.load_dotenv"), patch("notifier.ROOT", self.root), \
                    patch("sys.argv", ["notifier.py"]), \
                    patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "123:test", "COLLECTOR_DATA_DIR": str(self.root / "data")}), \
                    patch("notifier.TelegramClient.send") as send:
                self.assertEqual(notify_main(), 0)
            send.assert_not_called()

    def test_group_bot_mismatched_forwarded_and_noncommands_are_ignored(self):
        invalid = [None, {}, self.message("/startle"), self.message("hello"),
                   self.message("/start 33\nTOKEN=evil"), self.message(chat=-44)]
        for field, value in [("chat", {"id": 33, "type": "group"}),
                             ("from", {"id": 44, "is_bot": False}),
                             ("from", {"id": 33, "is_bot": True}),
                             ("forward_origin", {"type": "user"}),
                             ("forward_date", 123), ("text", None)]:
            invalid.append({**self.message(), field: value})
        before = self.env.read_text()
        for message in invalid:
            with self.subTest(message=message):
                self.assertIsNone(handle_subscription(message))
        self.assertEqual(self.env.read_text(), before)

    def test_parallel_signups_do_not_lose_recipients(self):
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(lambda chat: set_subscription(str(chat), True), range(100, 112)))
        self.assertEqual(set(recipient_chats()), {"11", "-22", *map(str, range(100, 112))})

    def test_missing_key_seeds_from_environment_and_invalid_ids_do_not_write(self):
        self.env.write_text("TOKEN=test-secret\n")
        with patch.dict(os.environ, {"TELEGRAM_CHAT_IDS": "11"}):
            handle_subscription(self.message())
        self.assertEqual(recipient_chats(), ["11", "33"])
        before = self.env.read_text()
        with self.assertRaises(ValueError):
            set_subscription("33\nTOKEN=evil", True)
        self.assertEqual(self.env.read_text(), before)

    def test_poll_persists_signup_offset_and_handles_replay_and_stop(self):
        client = Mock(bot="123")
        client.send.return_value = {"state": "unknown"}
        updates = [{"update_id": 20, "message": self.message()}]
        client.control.return_value = updates
        with closing(ReviewStore(self.root / "reviews.sqlite3")) as store:
            def confirm(*args):
                self.assertIn("33", recipient_chats())
                self.assertEqual(store.state("123")["next_update"], 21)
                return {"state": "unknown"}
            client.send.side_effect = confirm
            poll(self.root, "example", "11", "11", client, store)
            client.send.side_effect = None
            updates.append({"update_id": 21, "message": self.message("/stop")})
            poll(self.root, "example", "11", "11", client, store)
            self.assertNotIn("33", recipient_chats())
            self.assertEqual(client.send.call_count, 2)
        with closing(ReviewStore(self.root / "reviews.sqlite3")) as store:
            poll(self.root, "example", "11", "11", client, store)
            self.assertEqual(store.state("123")["next_update"], 22)
            self.assertEqual(client.send.call_count, 2)

    def test_failed_save_keeps_update_pending_without_confirmation(self):
        client = Mock(bot="123")
        client.control.return_value = [{"update_id": 20, "message": self.message()}]
        with closing(ReviewStore(self.root / "reviews.sqlite3")) as store, \
                patch("subscriptions.set_key", side_effect=OSError("read-only file")):
            with self.assertRaises(OSError):
                poll(self.root, "example", "11", "11", client, store)
            self.assertEqual(store.state("123")["next_update"], 0)
        client.send.assert_not_called()

    def test_review_delivery_uses_fresh_recipients_without_restart(self):
        client, store = Mock(bot="123"), Mock()
        store.rows.return_value = [{"target": "example", "story": "1", "decision": "approved"}]
        with patch.dict(os.environ, {"TELEGRAM_CHAT_IDS": "11,-22"}), \
                patch("review.candidates", return_value=[(0, "1", "alert")]), \
                patch("review.notify") as notify:
            handle_subscription(self.message())
            deliver_approved(self.root, "example", 24, client, store)
            self.assertEqual(notify.call_args.args[1], ["11", "-22", "33"])
            handle_subscription(self.message("/stop"))
            deliver_approved(self.root, "example", 24, client, store)
            self.assertEqual(notify.call_args.args[1], ["11", "-22"])

    def test_pipeline_sender_delivers_to_new_subscriber_and_respects_stop(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "123:test", "TELEGRAM_CHAT_IDS": "11,-22",
                                     "COLLECTOR_DATA_DIR": str(self.root / "data")}), \
                patch("notifier.load_dotenv"), patch("sys.argv", ["notifier.py"]), \
                patch("notifier.candidates", return_value=[(0, "story-1", "alert")]), \
                patch("notifier.TelegramClient.send", return_value={"state": "sent", "message_id": 42}) as send:
            handle_subscription(self.message())
            self.assertEqual(notify_main(), 0)
            self.assertEqual([call.args[0] for call in send.call_args_list], ["11", "-22", "33"])
            handle_subscription(self.message("/stop"))
            send.reset_mock()
            with patch("notifier.candidates", return_value=[(0, "story-2", "next alert")]):
                self.assertEqual(notify_main(), 0)
            self.assertEqual([call.args[0] for call in send.call_args_list], ["11", "-22"])


if __name__ == "__main__":
    unittest.main()
