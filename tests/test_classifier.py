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

    def test_server_error_exposes_reason_without_request_secrets(self):
        response = Mock()
        response.json.return_value = {"error": {"message": "Only one image supported. test-key https://example.com/private data:image/jpeg;base64,secret"}}
        text = server_error_message(response, "test-key")
        self.assertIn("Only one image supported", text)
        self.assertNotIn("test-key", text)
        self.assertNotIn("example.com", text)
        self.assertNotIn("base64,secret", text)


if __name__ == "__main__":
    unittest.main()
