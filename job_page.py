"""Fetch public job descriptions without Instagram cookies or model credentials."""
from datetime import date, datetime, timezone
import hashlib
from html import unescape
from html.parser import HTMLParser
import ipaddress
import json
from pathlib import Path
import re
import socket
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import requests

PAGE_VERSION = "job-page-v2"
MAX_BYTES = 1_500_000
MAX_TEXT = 8000


def canonical_url(url):
    for _ in range(3):
        parsed = urlsplit(str(url))
        if parsed.hostname in ("l.instagram.com", "lm.instagram.com"):
            target = dict(parse_qsl(parsed.query)).get("u")
            if target:
                url = target
                continue
        break
    if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Not a public HTTP job URL.")
    if parsed.port not in (None, 80, 443):
        raise ValueError("Unexpected job-page port.")
    query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in ("fbclid", "gclid", "igshid")]
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, urlencode(query), ""))


def check_public_host(url):
    """Reject local/private destinations, including redirect targets."""
    parsed = urlsplit(canonical_url(url))
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("Job URL points to a non-public address.")


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = []
        self.script = None
        self.scripts = []
        self.text = []
        self.h1 = []
        self.in_h1 = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self.script = [] if attrs.get("type", "").lower() == "application/ld+json" else None
        if tag in ("script", "style", "nav", "footer", "header", "form", "noscript"):
            self.skip.append(tag)
        if tag == "h1" and not self.skip:
            self.in_h1 = True

    def handle_endtag(self, tag):
        if tag == "script" and self.script is not None:
            self.scripts.append("".join(self.script))
            self.script = None
        if tag in self.skip:
            self.skip.remove(tag)
        if tag == "h1":
            self.in_h1 = False

    def handle_data(self, data):
        if self.script is not None:
            self.script.append(data)
        if not self.skip:
            value = " ".join(data.split())
            if value:
                self.text.append(value)
                if self.in_h1:
                    self.h1.append(value)


def plain_text(value):
    parser = PageParser()
    parser.feed(unescape(unescape(str(value))))
    return "\n".join(parser.text)


def postings(value):
    if isinstance(value, list):
        for child in value:
            yield from postings(child)
    elif isinstance(value, dict):
        types = value.get("@type", [])
        types = [types] if isinstance(types, str) else types
        if "JobPosting" in types:
            yield value
        else:
            for child in value.values():
                if isinstance(child, (list, dict)):
                    yield from postings(child)


def page_record(url, source, title, description, company=None, valid_through=None):
    if not isinstance(title, str) or not isinstance(description, str):
        return {"status": "unknown", "reason": "insufficient_job_text"}
    title = plain_text(title).strip()
    description = plain_text(description).strip()
    if not title or len(description) < 120:
        return {"status": "unknown", "reason": "insufficient_job_text"}
    availability = "unknown"
    if isinstance(valid_through, str):
        try:
            if "T" in valid_through:
                deadline = datetime.fromisoformat(valid_through.replace("Z", "+00:00"))
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                expired = deadline < datetime.now(timezone.utc)
            else:
                expired = date.fromisoformat(valid_through) < datetime.now(timezone.utc).date()
            if expired:
                availability = "expired"
        except ValueError:
            pass
    return {
        "status": "usable", "source": source, "url": url,
        "title": title[:300], "company": company if isinstance(company, str) else None,
        "description": description[:MAX_TEXT], "valid_through": valid_through,
        "availability": availability,
    }


def extract_html(html, url):
    parser = PageParser()
    parser.feed(html)
    text = "\n".join(parser.text)
    closed = re.search(r"\bthis (?:job|position)(?: posting)? is (?:closed|no longer available|filled)\b|\bno longer accepting applications for this (?:job|position)\b", text[:2500], re.I)
    jobs = []
    for script in parser.scripts:
        try:
            jobs.extend(postings(json.loads(script)))
        except (ValueError, TypeError):
            continue
    if len(jobs) > 1:
        return {"status": "unknown", "reason": "multiple_job_postings"}
    if jobs:
        job = jobs[0]
        # If a canonical job URL is supplied, don't use a different job's metadata.
        job_url = job.get("url")
        if isinstance(job_url, str) and canonical_url(urljoin(url, job_url)) != canonical_url(url):
            return {"status": "unknown", "reason": "job_url_mismatch"}
        org = job.get("hiringOrganization") or {}
        company = org.get("name") if isinstance(org, dict) else org
        result = page_record(url, "jobposting_jsonld", job.get("title", ""), job.get("description", ""), company, job.get("validThrough"))
        if closed:
            result["availability"] = "closed"
        return result
    title = " ".join(parser.h1)
    if closed:
        return {"status": "unknown", "reason": "explicitly_closed_job_page", "availability": "closed"}
    detail_path = re.search(r"/jobs?/[^/]+", urlsplit(url).path, re.I) or any(k.lower() in ("gh_jid", "jobid") for k, _ in parse_qsl(urlsplit(url).query))
    generic = re.search(r"^(?:careers?|open (?:roles|positions)|job search|search jobs|jobs|page not found)$", title, re.I)
    challenge = re.search(r"verify (?:you are|you're) human|access denied|captcha|job (?:is no longer available|not found)", text[:2000], re.I)
    # Conservative generic HTML fallback: a job-detail URL, heading, and substantial text.
    if not detail_path or not title or generic or challenge or len(text) < 300:
        return {"status": "unknown", "reason": "no_specific_job_description"}
    return page_record(url, "html_text", title, text)


class JobPageClient:
    def __init__(self, cache_dir, ttl_seconds=3600):
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.session = requests.Session()
        self.session.trust_env = False  # Don't send .netrc credentials to linked sites.

    def _get(self, url):
        for _ in range(4):
            check_public_host(url)
            with self.session.get(url, timeout=(5, 10), allow_redirects=False, stream=True,
                                  headers={"User-Agent": "sudo-scraper/0.1", "Accept": "text/html,application/json"}) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    url = canonical_url(urljoin(url, response.headers.get("Location", "")))
                    continue
                if response.status_code != 200:
                    return url, response.status_code, ""
                chunks, size = [], 0
                for chunk in response.iter_content(16384):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError("Job response exceeds the size limit.")
                    chunks.append(chunk)
                return url, 200, b"".join(chunks).decode("utf-8", errors="replace")
        raise ValueError("Too many job-page redirects.")

    def lookup(self, link, cached_only=False):
        try:
            url = canonical_url(link)
        except ValueError:
            return {"status": "unknown", "reason": "invalid_url"}
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.cache_dir / (key + ".json")
        if path.exists():
            cached = json.loads(path.read_text())
            ttl = self.ttl_seconds if cached.get("status") == "usable" else 60
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(cached["fetched_at"])).total_seconds()
            if cached.get("version") == PAGE_VERSION and age < ttl:
                return cached
        if cached_only:
            return {"status": "unknown", "reason": "not_fetched"}
        try:
            parsed = urlsplit(url)
            match = re.fullmatch(r"/([^/]+)/jobs/(\d+)/?", parsed.path)
            result = None
            if parsed.hostname in ("job-boards.greenhouse.io", "boards.greenhouse.io") and match:
                board, job_id = match.groups()
                api = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}"
                _, status, body = self._get(api)
                if status == 200:
                    job = json.loads(body)
                    if isinstance(job, dict) and str(job.get("id")) == job_id:
                        result = page_record(url, "greenhouse_api", job.get("title", ""), job.get("content", ""))
            if not result or result.get("status") != "usable":
                final_url, status, html = self._get(url)
                result = extract_html(html, final_url) if status == 200 else {"status": "unknown", "reason": f"http_{status}"}
        except (requests.RequestException, OSError, ValueError, TypeError):
            result = {"status": "unknown", "reason": "fetch_or_parse_failed"}
        result.setdefault("url", url)
        result["version"] = PAGE_VERSION
        result["content_hash"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        result["fetched_at"] = datetime.now(timezone.utc).isoformat()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(path)
        return result
