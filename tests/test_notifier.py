"""Offline delivery tests: fake Telegram responses, temporary databases only."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import requests

from notifier import DeliveryStore, TelegramClient, candidates, notify

NOW = datetime(2026, 10, 24, 12, tzinfo=timezone.utc).timestamp()
TOKEN = "123:fake_test_token"


class NotifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "deliveries.sqlite3"
        self.store = DeliveryStore(self.path)
        self.addCleanup(self.store.close)
        self.session = Mock()
        self.client = TelegramClient(TOKEN, self.session)
        self.record = {
            "story_id": "12345", "target": "zero2sudo", "status": "complete",
            "taken_at": "2026-10-24T11:00:00Z", "review_required": False,
            "decision": {"classification": "swe_internship", "company": "Example",
                         "role": "Software Engineering Intern"},
            "links": ["https://l.instagram.com/?u=https%3A%2F%2Fexample.com%2Fjobs%2F1%3Futm_source%3Dinsta"],
            "story_url": "https://www.instagram.com/stories/zero2sudo/12345/",
        }
        self.items = [(NOW - 3600, "12345", "SWE internship\nExample\nhttps://example.com/jobs/1")]

    def respond(self, code=200, body=None):
        self.session.post.return_value = Mock(status_code=code)
        self.session.post.return_value.json.return_value = (
            {"ok": True, "result": {"message_id": 42}} if body is None else body)

    def run_sender(self, chats=None, **kwargs):
        return notify(self.items, chats or ["11"], self.client, self.store,
                      clock=lambda: NOW, sleep=Mock(), **kwargs)

    def save_record(self, record=None):
        path = self.root / "classifications/zero2sudo/12345.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record or self.record))

    def test_success_is_durable_and_not_resent(self):
        self.respond()
        self.assertEqual(self.run_sender()["sent"], 1)
        with self.path.open("rb") as handle:
            self.assertNotIn(TOKEN.encode(), handle.read())
        reopened = DeliveryStore(self.path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.get("123", "11", "12345")["message_id"], 42)
        counts = notify(self.items, ["11"], self.client, reopened, clock=lambda: NOW)
        self.assertEqual(counts["already_sent"], 1)
        self.assertEqual(self.session.post.call_count, 1)
        payload = self.session.post.call_args.kwargs["json"]
        self.assertNotIn("parse_mode", payload)
        self.assertEqual(payload["chat_id"], "11")

    def test_recipient_failure_does_not_undo_other_recipient(self):
        self.session.post.side_effect = [
            Mock(status_code=200, json=lambda: {"ok": True, "result": {"message_id": 1}}),
            Mock(status_code=403, json=lambda: {"ok": False, "error_code": 403}),
        ]
        counts = self.run_sender(["11", "22"])
        self.assertEqual(counts, {"sent": 1, "rejected": 1})
        self.run_sender(["11", "22"])
        self.assertEqual(self.session.post.call_count, 2)

    def test_delivery_limit_counts_recipients(self):
        self.respond()
        self.assertEqual(self.run_sender(["11", "22"], limit=1)["sent"], 1)
        self.assertIsNone(self.store.get("123", "22", "12345"))

    def test_dry_run_never_sends_or_changes_state(self):
        counts = self.run_sender(dry_run=True)
        self.assertEqual(counts["pending"], 1)
        self.session.post.assert_not_called()
        self.assertEqual(self.store.rows("123"), [])

    def test_readonly_preview_does_not_create_database(self):
        path = self.root / "missing" / "preview.sqlite3"
        store = DeliveryStore(path, readonly=True)
        self.addCleanup(store.close)
        self.assertEqual(store.rows("123"), [])
        self.assertFalse(path.parent.exists())

    def test_rate_limit_persists_bot_wide_cooldown(self):
        self.respond(429, {"ok": False, "error_code": 429, "parameters": {"retry_after": 90}})
        self.assertEqual(self.run_sender(["11", "22"])["retry"], 1)
        self.assertEqual(self.store.get("123", "11", "12345")["next_attempt"], NOW + 90)
        self.assertEqual(self.run_sender(["22"])["cooldown"], 1)
        self.assertEqual(self.session.post.call_count, 1)
        self.respond()
        counts = notify(self.items, ["11"], self.client, self.store, clock=lambda: NOW + 91)
        self.assertEqual(counts["sent"], 1)
        self.assertEqual(self.store.get("123", "11", "12345")["attempts"], 2)

    def test_connect_timeout_can_retry_but_read_timeout_cannot(self):
        self.session.post.side_effect = requests.ConnectTimeout("secret URL " + TOKEN)
        self.assertEqual(self.run_sender()["retry"], 1)
        self.assertNotIn(TOKEN, str(self.store.rows("123")))
        self.session.post.side_effect = requests.ReadTimeout("secret URL " + TOKEN)
        self.assertEqual(self.run_sender(["22"])["cooldown"], 1)
        result = self.client.send("22", "test")
        self.assertEqual(result["state"], "unknown")
        self.assertNotIn(TOKEN, str(result))

    def test_ambiguous_delivery_is_held_until_manual_resolution(self):
        self.session.post.side_effect = requests.ReadTimeout()
        self.assertEqual(self.run_sender()["unknown"], 1)
        self.run_sender()
        self.assertEqual(self.session.post.call_count, 1)
        self.store.resolve("123", "11", "12345", "retry")
        self.session.post.side_effect = None
        self.respond()
        counts = notify(self.items, ["11"], self.client, self.store, clock=lambda: NOW + 10**9)
        self.assertEqual(counts["sent"], 1)

    def test_crash_after_begin_does_not_resend(self):
        self.store.begin("123", "11", "12345", NOW)
        self.assertEqual(self.run_sender()["sending"], 1)
        self.session.post.assert_not_called()
        self.store.resolve("123", "11", "12345", "sent")
        self.assertEqual(self.store.get("123", "11", "12345")["state"], "sent")
        with self.assertRaises(ValueError):
            self.store.resolve("123", "11", "12345", "retry")

    def test_new_bot_has_separate_delivery_state(self):
        self.store.begin("999", "11", "12345", NOW)
        self.respond()
        self.assertEqual(self.run_sender()["sent"], 1)

    def test_server_errors_and_malformed_success_are_ambiguous(self):
        for code, body in [(500, {"ok": False}), (200, {"ok": True, "result": {}}),
                           (302, {"ok": False}), (200, [])]:
            with self.subTest(code=code, body=body):
                self.respond(code, body)
                self.assertEqual(self.client.send("11", "test")["state"], "unknown")
                self.assertFalse(self.session.post.call_args.kwargs["allow_redirects"])

    def test_invalid_json_is_ambiguous(self):
        self.respond()
        self.session.post.return_value.json.side_effect = ValueError()
        self.assertEqual(self.client.send("11", "test")["state"], "unknown")

    def test_null_result_and_bad_rate_limit_parameters_are_safe(self):
        self.respond(200, {"ok": True, "result": None})
        self.assertEqual(self.client.send("11", "test")["state"], "unknown")
        for parameters in (None, [], {"retry_after": "bad"}, {"retry_after": float("inf")}):
            self.respond(429, {"ok": False, "error_code": 429, "parameters": parameters})
            self.assertEqual(self.client.send("11", "test")["retry_after"], 60)

    def test_sending_is_committed_before_network_request(self):
        def inspect_request(*args, **kwargs):
            reopened = DeliveryStore(self.path)
            try:
                self.assertEqual(reopened.get("123", "11", "12345")["state"], "sending")
            finally:
                reopened.close()
            return Mock(status_code=200, json=lambda: {"ok": True, "result": {"message_id": 9}})
        self.session.post.side_effect = inspect_request
        self.assertEqual(self.run_sender()["sent"], 1)

    def test_auth_failure_stops_batch(self):
        self.respond(401, {"ok": False, "error_code": 401})
        self.assertEqual(self.run_sender(["11", "22"])["rejected"], 1)
        self.assertEqual(self.session.post.call_count, 1)

    def test_candidates_use_final_decision_and_clean_links(self):
        self.record["model_decision"] = {"classification": "irrelevant"}
        self.save_record()
        result = candidates(self.root, "zero2sudo", 24, NOW)
        self.assertEqual(len(result), 1)
        self.assertIn("https://example.com/jobs/1", result[0][2])
        self.assertNotIn("l.instagram.com", result[0][2])
        self.assertIn("Story: https://www.instagram.com/stories/zero2sudo/12345/", result[0][2])

    def test_unsafe_or_old_results_are_excluded(self):
        changes = [
            {"status": "error"}, {"review_required": True}, {"review_required": None},
            {"decision": {"classification": "uncertain"}}, {"job_status": "closed"},
            {"job_status": "expired"}, {"links": []}, {"links": ["javascript:alert(1)"]},
            {"taken_at": "2026-10-20T00:00:00Z"}, {"taken_at": "2026-10-25T00:00:00Z"},
            {"taken_at": "2026-10-24T11:00:00"}, {"taken_at": "bad date"},
        ]
        for change in changes:
            with self.subTest(change=change):
                record = deepcopy(self.record)
                record.update(change)
                self.save_record(record)
                self.assertEqual(candidates(self.root, "zero2sudo", 24, NOW), [])

    def test_oversize_message_does_not_truncate_links(self):
        self.record["links"] = ["https://example.com/" + "x" * 4000]
        self.save_record()
        self.assertEqual(candidates(self.root, "zero2sudo", 24, NOW), [])

    def test_chat_discovery_only_returns_ids_and_types(self):
        self.respond(200, {"ok": True, "result": [
            {"message": {"chat": {"id": 11, "type": "private"}, "text": "private text"}},
            {"channel_post": {"chat": {"id": -100, "type": "channel"}}},
        ]})
        self.assertEqual(self.client.list_chats(), {"11": "private", "-100": "channel"})
        self.assertNotIn("offset", self.session.post.call_args.kwargs["json"])

    def test_discovery_error_does_not_expose_token(self):
        self.session.post.side_effect = requests.ConnectionError(TOKEN)
        with self.assertRaises(RuntimeError) as error:
            self.client.list_chats()
        self.assertNotIn(TOKEN, str(error.exception))

    def test_malformed_chat_discovery_is_sanitized(self):
        for body in ([], {"ok": True, "result": None}):
            self.respond(200, body)
            with self.assertRaises(RuntimeError):
                self.client.list_chats()
        self.respond(200, {"ok": True, "result": [None, {"message": None}]})
        self.assertEqual(self.client.list_chats(), {})


if __name__ == "__main__":
    unittest.main()
