"""Offline pipeline ordering, failure, locking, and scheduling tests."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pipeline import build_steps, main, pipeline_lock, preflight, run_once, run_pipeline


def options(**overrides):
    values = dict(skip_collect=False, skip_notify=False, mode="auto", provider=None,
                  model=None, classification_limit=None, notify_limit=20, max_age_hours=24)
    values.update(overrides)
    return SimpleNamespace(**values)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch("subscriptions.ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.steps = [("collect", ["python", "collector.py"]),
                      ("classify", ["python", "classifier.py"]),
                      ("notify", ["python", "notifier.py"])]

    def status(self):
        return json.loads((self.root / "last_run.json").read_text())

    def test_plan_uses_current_environment_auto_mode_and_no_login(self):
        steps = build_steps(options())
        self.assertEqual([name for name, _ in steps], ["collect", "classify", "notify"])
        for _, command in steps:
            self.assertEqual(command[0], sys.executable)
            self.assertTrue(Path(command[1]).is_absolute())
            self.assertNotIn("--login", command)
        self.assertIn("auto", steps[1][1])

    def test_plan_forwards_model_limits_and_age(self):
        steps = build_steps(options(skip_collect=True, provider="ollama", model="example",
                                    classification_limit=3, notify_limit=1, max_age_hours=72))
        self.assertEqual(steps[0][0], "classify")
        self.assertEqual(steps[0][1][-6:], ["--provider", "ollama", "--model", "example", "--limit", "3"])
        self.assertEqual(steps[1][1][-4:], ["--max-age-hours", "72", "--limit", "1"])
        self.assertEqual([s[0] for s in build_steps(options(skip_notify=True))], ["collect", "classify"])

    def test_success_runs_in_order_and_saves_status(self):
        runner = Mock(return_value=SimpleNamespace(returncode=0))
        record = run_once(self.steps, self.root, runner=runner)
        self.assertEqual(record["status"], "complete")
        self.assertEqual([call.args[0] for call in runner.call_args_list], [s[1] for s in self.steps])
        self.assertEqual([s["status"] for s in self.status()["stages"]], ["complete"] * 3)
        self.assertIsNotNone(self.status()["finished_at"])
        self.assertEqual(len((self.root / "events.jsonl").read_text().splitlines()), 8)
        self.assertEqual(runner.call_args.kwargs["timeout"], 3600)
        self.assertFalse(runner.call_args.kwargs["check"])
        self.assertNotIn("shell", runner.call_args.kwargs)

    def test_stage_failure_stops_later_stages(self):
        for failed_index in range(3):
            with self.subTest(failed_index=failed_index):
                runner = Mock(side_effect=[SimpleNamespace(returncode=0)] * failed_index
                              + [SimpleNamespace(returncode=1)])
                record = run_once(self.steps, self.root, runner=runner)
                self.assertEqual(record["status"], "failed")
                self.assertEqual(runner.call_count, failed_index + 1)
                self.assertEqual(record["stages"][-1]["exit_code"], 1)

    def test_instagram_error_is_a_distinct_stop(self):
        runner = Mock(return_value=SimpleNamespace(returncode=2))
        record = run_once(self.steps, self.root, runner=runner)
        self.assertEqual(record["status"], "instagram_stopped")
        runner.assert_called_once()

    def test_timeout_is_recorded_and_stops_pass(self):
        runner = Mock(side_effect=subprocess.TimeoutExpired(["python"], 10))
        record = run_once(self.steps, self.root, timeout=10, runner=runner)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(self.status()["stages"][0]["status"], "timed_out")
        self.assertEqual(self.status()["stages"][0]["exit_code"], 124)
        runner.assert_called_once()

    def test_process_start_failure_is_recorded_without_raw_exception(self):
        runner = Mock(side_effect=OSError("potentially sensitive details"))
        run_once(self.steps, self.root, runner=runner)
        self.assertEqual(self.status()["status"], "failed")
        self.assertNotIn("sensitive", (self.root / "events.jsonl").read_text())

    def test_interrupt_is_saved_and_propagated(self):
        with self.assertRaises(KeyboardInterrupt):
            run_once(self.steps, self.root, runner=Mock(side_effect=KeyboardInterrupt))
        self.assertEqual(self.status()["status"], "interrupted")
        self.assertEqual(self.status()["stages"][0]["exit_code"], 130)

    def test_running_state_is_committed_before_child_starts(self):
        def check_status(*args, **kwargs):
            self.assertEqual(self.status()["status"], "running")
            self.assertEqual(self.status()["stages"][-1]["status"], "running")
            return SimpleNamespace(returncode=0)
        run_once(self.steps, self.root, runner=check_status)

    def test_pipeline_lock_prevents_overlap_and_releases_after_exception(self):
        with pipeline_lock(self.root):
            with self.assertRaises(RuntimeError):
                with pipeline_lock(self.root):
                    self.fail("Overlapping lock acquired")
        with self.assertRaises(ValueError):
            with pipeline_lock(self.root):
                raise ValueError("test")
        with pipeline_lock(self.root):
            pass

    def test_one_shot_returns_nonzero_for_failure(self):
        for status, expected in (("complete", 0), ("failed", 1), ("instagram_stopped", 2)):
            self.assertEqual(run_pipeline(self.steps, self.root,
                                         once=Mock(return_value={"status": status})), expected)

    def test_watch_stops_on_instagram_error_without_sleeping(self):
        once, sleep = Mock(return_value={"status": "instagram_stopped"}), Mock()
        self.assertEqual(run_pipeline(self.steps, self.root, watch=True, once=once, sleep=sleep), 2)
        once.assert_called_once()
        sleep.assert_not_called()

    def test_watch_waits_for_remaining_interval_before_next_pass(self):
        once = Mock(side_effect=[{"status": "complete"}, {"status": "instagram_stopped"}])
        sleep = Mock()
        run_pipeline(self.steps, self.root, watch=True, interval=600, once=once,
                     clock=Mock(side_effect=[100, 140, 700]), sleep=sleep)
        sleep.assert_called_once_with(560)
        self.assertEqual(once.call_count, 2)

    def test_long_pass_never_causes_negative_sleep(self):
        once = Mock(side_effect=[{"status": "failed"}, {"status": "instagram_stopped"}])
        sleep = Mock()
        run_pipeline(self.steps, self.root, watch=True, interval=600, once=once,
                     clock=Mock(side_effect=[100, 800, 800]), sleep=sleep)
        sleep.assert_called_once_with(0)

    def test_preflight_requires_session_and_telegram_configuration(self):
        with patch.dict(os.environ, {"INSTAGRAM_USERNAME": "example"}, clear=True):
            with self.assertRaisesRegex(ValueError, "No saved Instagram session"):
                preflight(options(), self.root)
            (self.root / "session.json").write_text('{"collector_username":"example"}')
            with self.assertRaisesRegex(ValueError, "TELEGRAM_BOT_TOKEN"):
                preflight(options(), self.root)
            with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "123:test", "TELEGRAM_CHAT_IDS": "11,-22"}):
                preflight(options(), self.root)

    def test_preflight_skip_flags_do_not_require_unused_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            preflight(options(skip_collect=True, skip_notify=True), self.root)

    def test_preflight_accepts_empty_audience_waiting_for_signup(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "123:test"}, clear=True):
            preflight(options(skip_collect=True), self.root)

    def cli(self, *arguments):
        with patch.dict(os.environ, {"COLLECTOR_DATA_DIR": str(self.root / "data")}, clear=True), \
                patch("pipeline.load_dotenv"), patch.object(sys, "argv", ["pipeline.py", *arguments]):
            return main()

    def test_cli_dry_run_has_no_subprocess_or_disk_writes(self):
        with patch("pipeline.subprocess.run") as runner, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.cli("--dry-run"), 0)
        runner.assert_not_called()
        self.assertFalse((self.root / "data").exists())
        self.assertEqual(len(output.getvalue().splitlines()), 3)

    def test_cli_status_without_previous_run_does_not_write(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.cli("--status"), 0)
        self.assertEqual(json.loads(output.getvalue()), {})
        self.assertFalse((self.root / "data").exists())

    def test_cli_rejects_unsafe_intervals_and_conflicting_actions(self):
        for args in (("--interval-seconds", "1"), ("--stage-timeout", "nan"),
                     ("--watch", "--dry-run"), ("--classification-limit", "0")):
            with self.subTest(args=args), self.assertLogs("pipeline", level="ERROR"):
                self.assertEqual(self.cli(*args), 1)


if __name__ == "__main__":
    unittest.main()
