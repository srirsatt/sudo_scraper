"""Offline recovery tests; never log in to Instagram or send messages."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from instagrapi import Client
from collector import authenticate, collect_once, error_diagnostics, extract_links, read_json, save_session, save_story


def example_story():
    return SimpleNamespace(
        pk="123", media_type=1, thumbnail_url="https://example.com/story.jpg",
        model_dump=lambda **_: {"links": [{"webUri": "https://example.com/apply"}]},
    )


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.client = Mock()

        def download(url, filename, folder):
            path = folder / (filename + ".jpg")
            path.write_bytes(b"example downloaded media")
            return path

        self.client.story_download_by_url.side_effect = download

    def test_completed_story_is_not_downloaded_again(self):
        story = example_story()
        self.assertTrue(save_story(self.client, story, "example", self.root))
        self.assertFalse(save_story(self.client, story, "example", self.root))
        self.assertEqual(self.client.story_download_by_url.call_count, 1)

    def test_failed_download_preserves_links_and_can_resume(self):
        download = self.client.story_download_by_url.side_effect
        self.client.story_download_by_url.side_effect = OSError("offline")
        with self.assertRaises(OSError):
            save_story(self.client, example_story(), "example", self.root)
        path = self.root / "stories/example/123/story.json"
        record = read_json(path)
        self.assertFalse(record["complete"])
        self.assertEqual(record["links"], ["https://example.com/apply"])
        self.client.story_download_by_url.side_effect = download
        self.assertTrue(save_story(self.client, example_story(), "example", self.root))
        self.assertTrue(read_json(path)["complete"])

    def test_missing_media_is_downloaded_again(self):
        story = example_story()
        save_story(self.client, story, "example", self.root)
        (self.root / "stories/example/123/media.jpg").unlink()
        self.assertTrue(save_story(self.client, story, "example", self.root))

    def test_target_lookup_is_cached_across_runs(self):
        self.client.user_info_by_username_v1.return_value = SimpleNamespace(pk="42")
        self.client.user_stories_v1.return_value = []
        self.assertEqual(collect_once(self.client, "example", self.root), 0)
        self.assertEqual(collect_once(self.client, "example", self.root), 0)
        self.client.user_info_by_username_v1.assert_called_once_with("example")
        self.assertIn("last_success_at", read_json(self.root / "state.json"))

    def test_failed_fetch_does_not_record_success(self):
        self.client.user_info_by_username_v1.return_value = SimpleNamespace(pk="42")
        self.client.user_stories_v1.side_effect = RuntimeError("network unavailable")
        with self.assertRaises(RuntimeError):
            collect_once(self.client, "example", self.root)
        self.assertNotIn("last_success_at", read_json(self.root / "state.json"))

    def test_regular_run_never_password_logs_in(self):
        session = self.root / "session.json"
        session.write_text('{"collector_username": "example"}')
        self.client.username = "example"
        self.client.user_id = "42"
        authenticate(self.client, "example", session, login=False)
        self.client.login.assert_not_called()

    def test_real_client_settings_restore_authenticated_identity_offline(self):
        first = Client()
        first.authorization_data = {"ds_user_id": "42", "sessionid": "offline-test"}
        session = self.root / "session.json"
        save_session(first, "example", session)
        restored = Client()
        authenticate(restored, "example", session, login=False)
        self.assertEqual(str(restored.user_id), "42")
        self.assertEqual(restored.username, "example")

    def test_error_diagnostics_redact_secrets_and_skip_nested_payloads(self):
        client = Client()
        client.username = "example_user"
        client.password = "private-password"
        client.authorization_data = {"sessionid": "private-session-cookie"}
        client.last_json = {
            "status": "fail", "error_type": "needs_upgrade",
            "message": "Update Instagram example_user private-password private-session-cookie https://example.com/?token=secret example@example.com 123456",
            "challenge": {"url": "https://example.com/private-challenge"},
            "headers": {"Authorization": "secret-header"},
        }
        error = RuntimeError("Do not print this raw exception")
        error.response = SimpleNamespace(status_code=400)
        with patch.dict("os.environ", {"INSTAGRAM_PASSWORD": "private-password"}):
            details = error_diagnostics(client, error)
        rendered = str(details)
        for secret in ("private-password", "private-session-cookie", "private-challenge", "secret-header", "example_user", "123456", "example@example.com"):
            self.assertNotIn(secret, rendered)
        self.assertEqual(details["error_type"], "needs_upgrade")
        self.assertEqual(details["http_status"], 400)
        self.assertTrue(details["challenge_present"])

    def test_explicit_login_migrates_saved_profile_and_keeps_device(self):
        original = Client()
        session = self.root / "session.json"
        save_session(original, "example", session)
        restored = Client()
        with patch.dict("os.environ", {"INSTAGRAM_PASSWORD": "test-password"}):
            with patch.object(restored, "load_settings", wraps=restored.load_settings) as load:
                with patch.object(restored, "login", return_value=True):
                    authenticate(restored, "example", session, login=True)
        load.assert_called_once_with(session, override_app_version=True)
        self.assertEqual(restored.android_device_id, original.android_device_id)
        self.assertEqual(restored.private_transport, "curl")
        self.assertEqual(restored.session_retry_total, 0)

    def test_failed_first_login_preserves_device_identity(self):
        client = Client()
        session = self.root / "session.json"
        with patch.dict("os.environ", {"INSTAGRAM_PASSWORD": "test-password"}):
            with patch.object(client, "login", side_effect=RuntimeError("login failed")):
                with self.assertRaises(RuntimeError):
                    authenticate(client, "example", session, login=True)
        self.assertEqual(read_json(session)["uuids"]["android_device_id"], client.android_device_id)

    def test_links_include_stickers_without_duplicates(self):
        self.assertEqual(extract_links({
            "links": [{"webUri": "https://example.com/apply"}],
            "stickers": [{"story_link": {"url": "https://example.com/apply"}},
                         {"story_link": {"url": "https://example.com/other"}}],
        }), ["https://example.com/apply", "https://example.com/other"])


if __name__ == "__main__":
    unittest.main()
