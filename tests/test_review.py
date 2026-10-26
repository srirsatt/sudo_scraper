"""Offline Telegram review authorization, persistence, and delivery tests."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import requests

from collector import write_json
from notifier import DeliveryStore, TelegramClient, TelegramControlError, candidates, notify
from review import (ReviewStore, caption, clip, fingerprint, handle_callback,
                    load_records, markup, poll, send_cards)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "telegram").mkdir()
        self.store = ReviewStore(self.root / "telegram/reviews.sqlite3")
        self.addCleanup(self.store.close)
        self.now = time.time()
        self.record = {"story_id": "12345", "target": "example", "status": "complete",
            "taken_at": datetime.fromtimestamp(self.now - 60, timezone.utc).isoformat(),
            "classified_at": "old", "input_fingerprint": "input-v1", "review_required": True,
            "decision": {"classification": "uncertain", "company": "Example", "role": "AI Intern",
                         "evidence": ["AI Internship"], "reason": "Coding duties need review."},
            "links": ["https://example.com/jobs/1"],
            "story_url": "https://www.instagram.com/stories/example/12345/"}
        self.result = self.root / "classifications/example/12345.json"
        self.result.parent.mkdir(parents=True)
        write_json(self.result, self.record)
        folder = self.root / "stories/example/12345"
        folder.mkdir(parents=True)
        write_json(folder / "story.json", {"media_file": "media.jpg"})
        Image.new("RGB", (20, 30), "white").save(folder / "media.jpg")
        self.client = Mock()
        self.client.bot = "123"
        self.client.send_review.return_value = {"state": "sent", "message_id": 42}
        self.row = self.store.enqueue("123", self.record)
        self.store.delivery(self.row["id"], "sent", self.now, message_id=42)

    def query(self, action="a", **changes):
        query = {"id": "callback-1", "data": "r:" + self.row["id"] + ":" + action,
                 "from": {"id": 11}, "message": {"message_id": 42, "chat": {"id": 11, "type": "private"}}}
        query.update(changes)
        return query

    def handle(self, query):
        return handle_callback(query, self.root, "example", "11", "11", self.client, self.store)

    def ready(self):
        return candidates(self.root, "example", 24, self.now, bot="123")

    def test_approval_unlocks_held_result_and_persists(self):
        self.assertEqual(self.ready(), [])
        self.assertIn("Approved", self.handle(self.query()))
        self.assertEqual(len(self.ready()), 1)
        reopened = ReviewStore(self.root / "telegram/reviews.sqlite3")
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.get(self.row["id"])["decision"], "approved")
        self.assertEqual(json.loads(self.result.read_text()), self.record)

    def test_skip_stays_skipped_after_rerun(self):
        self.handle(self.query("s"))
        self.assertEqual(self.ready(), [])
        self.assertEqual(self.store.enqueue("123", self.record)["id"], self.row["id"])
        self.assertIn("Already skipped", self.handle(self.query()))
        self.assertEqual(self.store.get(self.row["id"])["decision"], "skipped")

    def test_unauthorized_chat_user_and_group_cannot_approve(self):
        queries = [self.query(**{"from": {"id": 22}}),
                   self.query(message={"message_id": 42, "chat": {"id": 22, "type": "private"}}),
                   self.query(message={"message_id": 42, "chat": {"id": 11, "type": "group"}})]
        for query in queries:
            self.assertIn("configured reviewer", self.handle(query))
            self.assertIsNone(self.store.get(self.row["id"])["decision"])

    def test_forged_callback_wrong_message_or_other_bot_is_rejected(self):
        for query in [self.query(data="r:" + "a" * 32 + ":a"), self.query(data="../../approve"),
                      self.query(message={"message_id": 99, "chat": {"id": 11, "type": "private"}})]:
            self.assertEqual(self.handle(query), "Unknown review card.")
        self.client.bot = "999"
        self.assertEqual(self.handle(self.query()), "Unknown review card.")
        self.assertIsNone(self.store.get(self.row["id"])["decision"])

    def test_stale_inputs_cannot_be_approved(self):
        changed = deepcopy(self.record)
        changed["input_fingerprint"] = "new-input"
        write_json(self.result, changed)
        self.assertIn("changed", self.handle(self.query()))
        self.assertIsNone(self.store.get(self.row["id"])["decision"])

    def test_input_changes_invalidate_old_approval(self):
        self.handle(self.query())
        self.record["links"] = ["https://example.com/jobs/2"]
        write_json(self.result, self.record)
        self.assertEqual(self.ready(), [])
        self.assertNotEqual(self.store.enqueue("123", self.record)["id"], self.row["id"])

    def test_classification_timestamp_does_not_invalidate_review(self):
        previous = fingerprint(self.record)
        self.record["classified_at"] = "new timestamp"
        self.assertEqual(fingerprint(self.record), previous)

    def test_approved_result_still_requires_age_link_and_open_job(self):
        self.handle(self.query())
        for change in ({"job_status": "closed"}, {"job_status": "expired"}, {"links": []},
                       {"taken_at": "2020-01-01T00:00:00Z"}):
            record = {**self.record, **change}
            # Even a matching approval cannot bypass the independent sender gates.
            row = self.store.enqueue("123", record)
            self.store.decide(row["id"], "approved", "11", 42)
            write_json(self.result, record)
            self.assertEqual(self.ready(), [])

    def test_review_approval_uses_existing_delivery_deduplication(self):
        self.handle(self.query())
        deliveries = DeliveryStore(self.root / "telegram/deliveries.sqlite3")
        self.addCleanup(deliveries.close)
        self.client.send.return_value = {"state": "sent", "message_id": 50}
        notify(self.ready(), ["22"], self.client, deliveries)
        notify(self.ready(), ["22"], self.client, deliveries)
        self.client.send.assert_called_once()

    def test_media_upload_is_not_repeated_for_existing_card(self):
        self.assertEqual(send_cards([self.record], self.root, "example", "11", self.client, self.store), 0)
        self.client.send_review.assert_not_called()

    def test_sending_is_committed_before_card_network_request(self):
        self.record["input_fingerprint"] = "second"
        def inspect(*args):
            row = self.store.find("123", self.record)
            self.assertEqual(row["delivery_state"], "sending")
            return {"state": "sent", "message_id": 43}
        self.client.send_review.side_effect = inspect
        self.assertEqual(send_cards([self.record], self.root, "example", "11", self.client, self.store), 1)
        self.assertEqual(self.store.find("123", self.record)["message_id"], 43)

    def test_card_read_timeout_is_held_and_can_be_resolved_by_actual_button(self):
        self.store.delivery(self.row["id"], "unknown", self.now)
        self.assertEqual(send_cards([self.record], self.root, "example", "11", self.client, self.store), 0)
        self.assertIn("Approved", self.handle(self.query()))
        self.assertEqual(self.store.get(self.row["id"])["message_id"], 42)

    def test_retry_requires_explicit_unresolved_card(self):
        with self.assertRaises(ValueError):
            self.store.retry("123", self.row["id"])
        self.store.delivery(self.row["id"], "unknown", self.now)
        self.store.retry("123", self.row["id"])
        self.assertEqual(self.store.get(self.row["id"])["delivery_state"], "pending")

    def test_rate_limit_persists_and_stops_review_batch(self):
        self.record["input_fingerprint"] = "second"
        self.client.send_review.return_value = {"state": "retry", "error": "rate_limited", "retry_after": 90}
        send_cards([self.record], self.root, "example", "11", self.client, self.store)
        self.assertGreater(self.store.state("123")["cooldown"], self.now)
        self.assertEqual(send_cards([self.record], self.root, "example", "11", self.client, self.store), 0)

    def test_dry_run_does_not_enqueue_or_send(self):
        self.record["input_fingerprint"] = "second"
        self.assertEqual(send_cards([self.record], self.root, "example", "11", self.client, self.store, dry_run=True), 1)
        self.assertIsNone(self.store.find("123", self.record))
        self.client.send_review.assert_not_called()

    def test_poll_persists_offset_after_decision_and_repeat_is_safe(self):
        self.client.control.return_value = [{"update_id": 20, "callback_query": self.query()}]
        poll(self.root, "example", "11", "11", self.client, self.store)
        self.assertEqual(self.store.state("123")["next_update"], 21)
        self.assertEqual(self.store.get(self.row["id"])["decision"], "approved")
        poll(self.root, "example", "11", "11", self.client, self.store)
        self.assertEqual(self.store.get(self.row["id"])["decision"], "approved")

    def test_acknowledgment_failure_does_not_lose_saved_decision(self):
        self.client.control.side_effect = TelegramControlError()
        self.assertIn("Approved", self.handle(self.query()))
        self.assertEqual(self.store.get(self.row["id"])["decision"], "approved")

    def test_queue_filters_old_closed_irrelevant_and_unheld_records(self):
        self.assertEqual(len(load_records(self.root, "example", 24, self.now)), 1)
        for change in ({"job_status": "closed"}, {"review_required": False},
                       {"taken_at": "2020-01-01T00:00:00Z"},
                       {"decision": {"classification": "irrelevant"}}):
            write_json(self.result, {**self.record, **change})
            self.assertEqual(load_records(self.root, "example", 24, self.now), [])

    def test_caption_and_callback_data_fit_telegram_limits(self):
        self.record["decision"]["reason"] = "🚀" * 2000
        self.assertLessEqual(len(caption(self.record).encode("utf-16-le")) // 2, 900)
        buttons = markup(self.record, self.row["id"])["inline_keyboard"][0]
        self.assertTrue(all(len(button["callback_data"].encode()) <= 64 for button in buttons))
        self.assertEqual(clip("🚀", 1), "")

    def test_control_failures_do_not_expose_token_and_preserve_retry_after(self):
        session = Mock()
        client = TelegramClient("123:private_token", session)
        session.post.side_effect = requests.ReadTimeout("123:private_token")
        with self.assertRaises(TelegramControlError) as error:
            client.control("getUpdates", {})
        self.assertNotIn("private_token", str(error.exception))
        session.post.side_effect = None
        session.post.return_value.status_code = 429
        session.post.return_value.json.return_value = {"ok": False, "error_code": 429, "parameters": {"retry_after": 70}}
        with self.assertRaises(TelegramControlError) as error:
            client.control("getUpdates", {})
        self.assertEqual(error.exception.retry_after, 70)

    def test_real_image_is_uploaded_as_multipart_photo_with_buttons(self):
        session = Mock()
        session.post.return_value.status_code = 200
        session.post.return_value.json.return_value = {"ok": True, "result": {"message_id": 8}}
        client = TelegramClient("123:test_token", session)
        path = self.root / "stories/example/12345/media.jpg"
        result = client.send_review("11", "Review", markup(self.record, self.row["id"]), path)
        self.assertEqual(result["state"], "sent")
        kwargs = session.post.call_args.kwargs
        self.assertIn("photo", kwargs["files"])
        self.assertEqual(json.loads(kwargs["data"]["reply_markup"])["inline_keyboard"][0][0]["text"], "Approve")


if __name__ == "__main__":
    unittest.main()
