"""Offline pipeline/HTTP contract tests, not a benchmark of model accuracy."""
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from classifier import apply_eligibility_policy, classify_saved, load_stories, nearby_images
from collector import read_json, write_json
from llama import Decision, ModelConfig, ModelOutputError, ModelUnavailable, VisionClassifier, parse_decision, server_error_message
from media import story_images
import imageio_ffmpeg
import requests
import base64
from dataclasses import replace
from io import BytesIO


def internship():
    return Decision(
        classification="swe_internship", company="Example", role="SWE Intern",
        evidence=["Software Engineering Intern applications open"],
        reason="The image advertises an actual software engineering internship opening.",
    )


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = Mock()
        self.model.config = ModelConfig("http://127.0.0.1:11434/v1", "test-vision")
        self.model.classify.return_value = internship()

    def story(self, story_id="123", minute=0, video=False):
        folder = self.root / "stories/example" / story_id
        folder.mkdir(parents=True, exist_ok=True)
        filename = "media.mp4" if video else "media.jpg"
        if video:
            (folder / filename).write_bytes(b"fake video, never decoded by this version")
        else:
            Image.new("RGB", (20, 30), "white").save(folder / filename)
        write_json(folder / "story.json", {
            "story_id": story_id, "target": "example", "complete": True,
            "media_file": filename, "links": ["https://example.com/apply"],
            "metadata": {"taken_at": f"2026-10-16T12:{minute:02d}:00+00:00"},
        })
        return folder

    def test_reads_collector_records_and_preserves_actual_links(self):
        self.story()
        counts = classify_saved(self.root, "example", self.model)
        record = read_json(self.root / "classifications/example/123.json")
        self.assertEqual(counts["swe_internship"], 1)
        self.assertEqual(record["links"], ["https://example.com/apply"])
        self.assertEqual(record["status"], "complete")
        self.assertFalse(record["review_required"])

    def test_product_manager_intern_is_not_accepted_as_swe(self):
        for company in ("Duolingo", "NVIDIA"):
            with self.subTest(company=company):
                raw = Decision(
                    classification="swe_internship", company=company,
                    role="Product Manager, MBA Intern", evidence=["Product Manager, MBA Intern"],
                    reason="Incorrect model positive from the reviewed stories.",
                )
                final, notes = apply_eligibility_policy(raw)
                self.assertEqual(final.classification, "irrelevant")
                self.assertTrue(notes)

    def test_engineer_i_without_internship_evidence_needs_review(self):
        raw = Decision(
            classification="swe_internship", company="Amazon",
            role="Post-Silicon Systems Validation Engineer I",
            evidence=["Post-Silicon Systems Validation Engineer I", "Annapurna Labs"],
            reason="Incorrect model positive from the reviewed stories.",
        )
        final, notes = apply_eligibility_policy(raw)
        self.assertEqual(final.classification, "uncertain")
        self.assertTrue(any("internship" in n for n in notes))

    def test_explicit_software_internship_still_passes_policy(self):
        final, notes = apply_eligibility_policy(internship())
        self.assertEqual(final.classification, "swe_internship")
        self.assertEqual(notes, [])

    def test_fulltime_software_engineer_is_not_assumed_to_be_intern(self):
        raw = Decision(
            classification="swe_internship", company="Example", role="Software Engineer I",
            evidence=["Software Engineer I", "Entry level"], reason="An entry-level opening.",
        )
        self.assertEqual(apply_eligibility_policy(raw)[0].classification, "uncertain")

    def test_cache_skips_unchanged_inputs(self):
        self.story()
        classify_saved(self.root, "example", self.model)
        self.assertEqual(classify_saved(self.root, "example", self.model)["cached"], 1)
        self.model.classify.assert_called_once()

    def test_new_neighbor_reclassifies_previous_story(self):
        self.story()
        classify_saved(self.root, "example", self.model)
        self.story("124", minute=1)
        classify_saved(self.root, "example", self.model)
        self.assertEqual(self.model.classify.call_count, 3)

    def test_far_apart_stories_do_not_supply_context(self):
        self.story()
        self.story("124", minute=45)
        stories = load_stories(self.root, "example")
        self.assertEqual(nearby_images(stories, 0), [])

    def test_fast_mode_omits_neighbors_and_has_separate_cache(self):
        self.story()
        self.story("124", minute=1)
        classify_saved(self.root, "example", self.model, limit=1)
        self.assertEqual(len(self.model.classify.call_args.args[1]), 1)
        self.model.config = replace(self.model.config, mode="fast")
        classify_saved(self.root, "example", self.model, limit=1)
        self.assertEqual(self.model.classify.call_args.args[1], [])
        self.assertEqual(self.model.classify.call_count, 2)
        record = read_json(self.root / "classifications/example/123.json")
        self.assertEqual(record["mode"], "fast")
        self.assertEqual(record["context_story_ids"], [])

    def test_fast_images_are_resized_and_context_is_not_sent(self):
        folder = self.story()
        self.story("124", minute=1)
        Image.new("RGB", (720, 1280), "white").save(folder / "media.jpg")
        model = VisionClassifier(replace(self.model.config, mode="fast"))
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": internship().model_dump_json()}}]}
        stories = load_stories(self.root, "example")
        with patch("llama.requests.post", return_value=response) as post:
            model.classify(stories[0], [stories[1]])
        content = post.call_args.kwargs["json"]["messages"][1]["content"]
        images = [p for p in content if p["type"] == "image_url"]
        self.assertEqual(len(images), 1)
        data = base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1])
        with Image.open(BytesIO(data)) as image:
            self.assertEqual(max(image.size), 1024)

    def test_bad_output_is_retryable_and_never_marked_irrelevant(self):
        self.story()
        self.model.classify.side_effect = ModelOutputError("invalid JSON")
        self.assertEqual(classify_saved(self.root, "example", self.model)["errors"], 1)
        path = self.root / "classifications/example/123.json"
        self.assertEqual(read_json(path)["status"], "error")
        self.assertNotIn("decision", read_json(path))
        self.model.classify.side_effect = None
        classify_saved(self.root, "example", self.model)
        self.assertEqual(read_json(path)["status"], "complete")

    def test_video_positive_uses_model_and_keeps_coverage_warning(self):
        self.story(video=True)
        classify_saved(self.root, "example", self.model)
        result = read_json(self.root / "classifications/example/123.json")
        self.assertEqual(result["decision"]["classification"], "swe_internship")
        self.assertEqual(result["media_coverage"], "sampled_frames_no_audio")
        self.assertTrue(result["review_required"])
        self.model.classify.assert_called_once()
        self.assertEqual(result["model_decision"]["classification"], "swe_internship")

    def test_video_negative_is_uncertain_because_audio_is_unreviewed(self):
        self.story(video=True)
        self.model.classify.return_value = Decision(
            classification="irrelevant", company=None, role=None,
            evidence=[], reason="No visible job posting.",
        )
        classify_saved(self.root, "example", self.model)
        result = read_json(self.root / "classifications/example/123.json")
        self.assertEqual(result["decision"]["classification"], "uncertain")

    def test_heic_and_real_video_frames_decode(self):
        folder = self.story()
        heic = folder / "photo.heic"
        Image.new("RGB", (64, 64), "white").save(heic)
        self.assertEqual(len(story_images({"media_path": heic, "metadata": {}})), 1)
        video = folder / "video.mp4"
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin",
            "-f", "lavfi", "-i", "color=c=white:s=64x64:d=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
        ], check=True, capture_output=True, timeout=30)
        story = {"media_path": video, "metadata": {"video_duration": 2}}
        self.assertEqual(len(story_images(story)), 3)
        self.assertEqual(len(story_images(story, context=True)), 1)
        self.assertEqual(len(story_images(story, mode="fast")), 1)

    def test_dry_run_does_not_call_model_or_write_results(self):
        self.story()
        classify_saved(self.root, "example", self.model, dry_run=True)
        self.assertFalse((self.root / "classifications").exists())
        self.model.classify.assert_not_called()

    def test_unavailable_server_stops_batch_and_invalidates_old_decision(self):
        self.story()
        self.story("124", minute=1)
        classify_saved(self.root, "example", self.model)
        self.model.classify.reset_mock()
        self.model.classify.side_effect = ModelUnavailable("server down")
        with self.assertRaises(ModelUnavailable):
            classify_saved(self.root, "example", self.model, force=True)
        self.model.classify.assert_called_once()
        result = read_json(self.root / "classifications/example/123.json")
        self.assertEqual(result["status"], "error")
        self.assertNotIn("decision", result)

    def test_http_payload_has_inline_images_and_optional_auth(self):
        self.story()
        model = VisionClassifier(ModelConfig("http://localhost:11434/v1", "test-vision", "test-key"))
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": internship().model_dump_json()}}]}
        with patch("llama.requests.post", return_value=response) as post:
            result = model.classify(load_stories(self.root, "example")[0], [])
        self.assertEqual(result.classification, "swe_internship")
        self.assertEqual(post.call_args.args[0], "http://localhost:11434/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")
        image = post.call_args.kwargs["json"]["messages"][1]["content"][1]
        self.assertTrue(image["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_schema_rejects_hallucinated_urls_and_empty_positive_evidence(self):
        data = internship().model_dump()
        data["application_url"] = "https://invented.example.com"
        with self.assertRaises(ModelOutputError):
            parse_decision(json.dumps(data))
        data.pop("application_url")
        data["evidence"] = []
        with self.assertRaises(ModelOutputError):
            parse_decision(json.dumps(data))

    def test_native_ollama_requests_large_context_and_decodes_response(self):
        self.story()
        model = VisionClassifier(ModelConfig(
            "http://localhost:11434/v1", "test-vision", api_style="ollama", context_length=16384,
        ))
        response = Mock(status_code=200)
        response.json.return_value = {
            "done_reason": "stop", "message": {"content": internship().model_dump_json()},
            "prompt_eval_count": 2000, "eval_count": 100, "prompt_eval_duration": 2_000_000_000,
        }
        with patch("llama.requests.post", return_value=response) as post:
            result = model.classify(load_stories(self.root, "example")[0], [])
        self.assertEqual(post.call_args.args[0], "http://localhost:11434/api/chat")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["options"]["num_ctx"], 16384)
        self.assertIs(payload["think"], False)
        self.assertEqual(len(payload["messages"][1]["images"]), 1)
        self.assertNotIn("data:image", payload["messages"][1]["images"][0])
        self.assertEqual(result.classification, "swe_internship")
        self.assertEqual(model.last_metrics["input_tokens"], 2000)
        self.assertEqual(model.last_metrics["prompt_eval_seconds"], 2)
        self.assertEqual(model.last_metrics["input_images"], 1)

    def test_server_error_exposes_reason_without_request_secrets(self):
        response = Mock()
        response.json.return_value = {"error": {"message": "Only one image supported. test-key https://example.com/private data:image/jpeg;base64,secret"}}
        text = server_error_message(response, "test-key")
        self.assertIn("Only one image supported", text)
        self.assertNotIn("test-key", text)
        self.assertNotIn("example.com", text)
        self.assertNotIn("base64,secret", text)

    def test_timeout_is_distinguished_from_connection_failure(self):
        self.story()
        model = VisionClassifier(ModelConfig("http://localhost:11434/v1", "test-vision", timeout=120))
        for error, expected in ((requests.ReadTimeout(), "read_timeout"),
                                (requests.ConnectTimeout(), "connect_timeout"),
                                (requests.ConnectionError(), "connection_error")):
            with self.subTest(kind=expected):
                with patch("llama.requests.post", side_effect=error):
                    with self.assertRaises(ModelUnavailable) as caught:
                        model.classify(load_stories(self.root, "example")[0], [])
                self.assertEqual(caught.exception.kind, expected)

    def test_fireworks_preset_ignores_stale_local_settings_and_uses_key(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "fireworks", "FIREWORKS_API_KEY": "fireworks-test-key",
            "FIREWORKS_MODEL": "accounts/fireworks/models/test-vision",
            "LLM_BASE_URL": "http://localhost:11434/v1", "LLM_API_STYLE": "ollama",
            "LLM_MODEL": "qwen3-vl:8b",
        }, clear=True):
            config = ModelConfig.from_env()
        self.assertEqual(config.base_url, "https://api.fireworks.ai/inference/v1")
        self.assertEqual(config.api_style, "openai")
        self.assertEqual(config.model, "accounts/fireworks/models/test-vision")
        self.assertNotIn("fireworks-test-key", repr(config))
        self.story()
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": internship().model_dump_json()}}]}
        with patch("llama.requests.post", return_value=response) as post:
            VisionClassifier(config).classify(load_stories(self.root, "example")[0], [])
        self.assertEqual(post.call_args.args[0], "https://api.fireworks.ai/inference/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer fireworks-test-key")
        payload = post.call_args.kwargs["json"]
        self.assertNotIn("options", payload)
        self.assertEqual(payload["response_format"]["type"], "json_schema")

    def test_fireworks_missing_key_or_model_fails_before_network(self):
        cases = (
            {"LLM_PROVIDER": "fireworks"},
            {"LLM_PROVIDER": "fireworks", "FIREWORKS_API_KEY": "test"},
            {"LLM_PROVIDER": "fireworks", "FIREWORKS_API_KEY": "test", "FIREWORKS_MODEL": "qwen3-vl:8b"},
        )
        with patch("llama.requests.post") as post:
            for env in cases:
                with self.subTest(env_keys=sorted(env)):
                    with patch.dict("os.environ", env, clear=True):
                        with self.assertRaises(ValueError):
                            ModelConfig.from_env()
            post.assert_not_called()

    def test_local_preset_never_inherits_cloud_credentials(self):
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "ollama", "LLM_API_KEY": "generic-cloud-key",
            "FIREWORKS_API_KEY": "fireworks-key",
        }, clear=True):
            config = ModelConfig.from_env()
        self.assertEqual(config.api_key, "")
        self.assertEqual(config.timeout, 300)
        self.assertFalse(config.ollama_think)
        self.assertEqual(config.model, "qwen3-vl:8b-instruct")

    def test_truncated_native_output_retains_metrics_and_is_retryable(self):
        self.story()
        config = ModelConfig("http://localhost:11434/v1", "test-vision", api_style="ollama")
        model = VisionClassifier(config)
        response = Mock(status_code=200)
        response.json.return_value = {
            "done_reason": "length", "eval_count": 1024,
            "message": {"content": "", "thinking": "hidden reasoning, not saved"},
        }
        with patch("llama.requests.post", return_value=response):
            counts = classify_saved(self.root, "example", model)
        self.assertEqual(counts["errors"], 1)
        result = read_json(self.root / "classifications/example/123.json")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["model_metrics"]["output_tokens"], 1024)
        self.assertGreater(result["model_metrics"]["thinking_chars"], 0)
        self.assertIn("reasoning despite think=false", result["error"])
        self.assertNotIn("hidden reasoning", json.dumps(result))

    def test_legacy_configs_remain_compatible(self):
        with patch.dict("os.environ", {"LLM_MODEL": "gemma3:4b"}, clear=True):
            self.assertEqual(ModelConfig.from_env().api_style, "ollama")
        with patch.dict("os.environ", {
            "LLM_API_STYLE": "openai", "LLM_MODEL": "vision-model",
            "LLM_BASE_URL": "https://example.com/v1", "LLM_API_KEY": "test-key",
        }, clear=True):
            self.assertEqual(ModelConfig.from_env().provider, "custom")

    def test_openai_preset_and_completion_budget(self):
        self.story()
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-openai-key",
            "OPENAI_MODEL": "test-vision-model", "LLM_BASE_URL": "http://localhost:11434/v1",
        }, clear=True):
            config = ModelConfig.from_env()
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": internship().model_dump_json()}}]}
        with patch("llama.requests.post", return_value=response) as post:
            VisionClassifier(config).classify(load_stories(self.root, "example")[0], [])
        self.assertEqual(post.call_args.args[0], "https://api.openai.com/v1/chat/completions")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["max_completion_tokens"], config.max_output_tokens)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("temperature", payload)
        self.assertFalse(payload["store"])

    def test_auth_and_throttle_errors_stop_without_retry_or_fallback(self):
        self.story()
        model = VisionClassifier(ModelConfig("https://api.fireworks.ai/inference/v1", "vision", "test-key", provider="fireworks"))
        for status, kind in ((401, "authentication_error"), (429, "rate_limited")):
            with self.subTest(status=status):
                response = Mock(status_code=status)
                response.json.return_value = {"error": {"message": "test-key rejected"}}
                with patch("llama.requests.post", return_value=response) as post:
                    with self.assertRaises(ModelUnavailable) as caught:
                        model.classify(load_stories(self.root, "example")[0], [])
                post.assert_called_once()
                self.assertEqual(caught.exception.kind, kind)
                self.assertNotIn("test-key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
