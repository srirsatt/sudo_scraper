"""Offline extraction and fetch/cache tests; no external HTTP requests."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from job_page import JobPageClient, canonical_url, extract_html, page_record

DESCRIPTION = "Build software services with Python and collaborate with software engineers. This software engineering internship is open to current university students. " * 3


class JobPageTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = JobPageClient(Path(self.temp.name))

    def html(self, jobs):
        return '<script type="application/ld+json">' + json.dumps(jobs) + '</script>'

    def job(self, **extra):
        return {"@type": "JobPosting", "title": "Software Engineering Intern", "description": DESCRIPTION,
                "hiringOrganization": {"name": "Example"}, **extra}

    def test_tracking_is_removed_but_job_identifiers_are_preserved(self):
        url = "https://example.com/careers?gh_jid=123&utm_source=instagram&fbclid=tracking"
        self.assertEqual(canonical_url(url), "https://example.com/careers?gh_jid=123")
        self.assertEqual(canonical_url("https://l.instagram.com/?u=https%3A%2F%2Fexample.com%2Fjobs%2F123"), "https://example.com/jobs/123")

    def test_jobposting_graph_is_extracted_and_expiry_recorded(self):
        result = extract_html(self.html({"@graph": [self.job(validThrough="2020-01-01")]}), "https://example.com/jobs/123")
        self.assertEqual(result["status"], "usable")
        self.assertEqual(result["company"], "Example")
        self.assertEqual(result["availability"], "expired")
        self.assertEqual(result["source"], "jobposting_jsonld")

    def test_future_deadline_does_not_guarantee_job_is_open(self):
        result = page_record("https://example.com/jobs/123", "test", "SWE Intern", DESCRIPTION, valid_through="2099-01-01")
        self.assertEqual(result["availability"], "unknown")

    def test_explicit_closed_message_is_retained_despite_old_job_metadata(self):
        html = '<p>This position is closed.</p>' + self.html(self.job())
        result = extract_html(html, "https://example.com/jobs/123")
        self.assertEqual(result["status"], "usable")
        self.assertEqual(result["availability"], "closed")

    def test_missing_title_is_not_usable(self):
        result = extract_html(self.html(self.job(title=None)), "https://example.com/jobs/123")
        self.assertEqual(result["status"], "unknown")

    def test_listing_pages_and_other_jobs_are_not_used(self):
        url = "https://example.com/jobs/123"
        self.assertEqual(extract_html(self.html([self.job(), self.job()]), url)["reason"], "multiple_job_postings")
        self.assertEqual(extract_html(self.html(self.job(url="https://example.com/jobs/999")), url)["reason"], "job_url_mismatch")

    def test_specific_html_job_can_be_extracted_without_scripts(self):
        result = extract_html('<nav>Ignore navigation</nav><h1>Software Engineering Intern</h1><main>' + DESCRIPTION + '</main>', "https://example.com/jobs/123")
        self.assertEqual(result["status"], "usable")
        self.assertEqual(result["source"], "html_text")
        self.assertNotIn("Ignore navigation", result["description"])

    def test_javascript_shell_and_challenge_are_unknown(self):
        url = "https://example.com/jobs/123"
        self.assertEqual(extract_html('<div id="app"></div><script>load()</script>', url)["status"], "unknown")
        self.assertEqual(extract_html('<h1>Security check</h1>Verify you are human ' + DESCRIPTION, url)["status"], "unknown")

    def test_greenhouse_api_and_canonical_cache(self):
        body = json.dumps({"id": 123, "title": "SWE Intern", "content": '<p>' + DESCRIPTION + '</p>'})
        with patch.object(self.client, "_get", return_value=("api-url", 200, body)) as get:
            first = self.client.lookup("https://job-boards.greenhouse.io/example/jobs/123?utm_source=instagram")
            second = self.client.lookup("https://job-boards.greenhouse.io/example/jobs/123?fbclid=other")
        get.assert_called_once_with("https://boards-api.greenhouse.io/v1/boards/example/jobs/123")
        self.assertEqual(first["source"], "greenhouse_api")
        self.assertEqual(first["content_hash"], second["content_hash"])

    def test_404_is_not_interpreted_as_closed(self):
        with patch.object(self.client, "_get", return_value=("https://example.com/jobs/123", 404, "")):
            result = self.client.lookup("https://example.com/jobs/123")
        self.assertEqual(result["status"], "unknown")
        self.assertNotEqual(result.get("availability"), "expired")

    def test_dry_lookup_never_fetches(self):
        with patch.object(self.client, "_get") as get:
            result = self.client.lookup("https://example.com/jobs/123", cached_only=True)
        get.assert_not_called()
        self.assertEqual(result["reason"], "not_fetched")

    def test_private_addresses_and_redirects_are_rejected(self):
        public = [(2, 1, 6, "", ("93.184.216.34", 443))]
        private = [(2, 1, 6, "", ("127.0.0.1", 80))]
        with patch("job_page.socket.getaddrinfo", return_value=private):
            with patch.object(self.client.session, "get") as get:
                with self.assertRaises(ValueError):
                    self.client._get("http://localhost/jobs/123")
                get.assert_not_called()
        response = Mock(status_code=302, headers={"Location": "http://localhost/private"})
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        with patch("job_page.socket.getaddrinfo", side_effect=[public, private]):
            with patch.object(self.client.session, "get", return_value=context) as get:
                with self.assertRaises(ValueError):
                    self.client._get("https://example.com/jobs/123")
                get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
