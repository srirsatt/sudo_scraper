# Sudo scraper

Collect zero2sudo's available Instagram stories locally, for a future SWE internship
classifier and Telegram notifier.
The classifier stage now reads those saved stories and writes decisions locally;
Telegram delivery is still pending.
It uses the unofficial `instagrapi` client; Instagram can restrict access, and neither
complete coverage nor a ten-minute detection interval is guaranteed.

## Local setup

Use Python 3.10 or newer on macOS/Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`: set **your** `INSTAGRAM_USERNAME` and confirm `INSTAGRAM_TARGET` is the
creator's exact handle. Leave `INSTAGRAM_PASSWORD` blank to enter it at the prompt,
or set it in `.env` (quote passwords containing spaces or `#`). Your account must
be able to view the target's stories normally.

```sh
python collector.py --login
```

This explicitly logs in, prompts for your password if needed and for a two-factor
code if Instagram requests one, saves a local session, then collects available
stories. Challenges other than the supported two-factor flow may require manual
resolution in Instagram. Don't repeatedly rerun login when Instagram blocks it.

If login returns `UnknownError`, the collector now prints an `Instagram diagnostics`
JSON line containing the HTTP status, error type, message, and challenge/2FA flags.
It redacts known credentials, cookies, URLs, email addresses, and long numbers, and
doesn't print nested responses or headers. Review this line before sharing it.
HTTP 400 by itself doesn't identify a bad password, throttling, or an outdated
private login flow. Check the actual message and any login approval in the official
Instagram app before another attempt. Failed logins also save device settings, so
the next explicit attempt reuses the same device identity. Retry options apply to
login as well as collection.

For `error_type: needs_upgrade`, update the dependencies and explicitly log in:

```sh
python -m pip install -r requirements.txt
python collector.py --login
```

The project uses instagrapi 3.0.25's default CAA login and curl/HTTP2 private
transport. `--login` refreshes saved app-version and Bloks-hash fields using the
library's supported profile, preserving device identifiers and cookies. Do not
delete `data/session.json` to perform this migration. This addresses the old app
profile/login implementation; Instagram may still require account verification or
reject access for another reason. See the
[upstream migration guide](https://subzeroid.github.io/instagrapi/latest/usage-guide/login-migration/).

Subsequent collection passes reuse the session without a password login:

```sh
python collector.py
```

The collector needs no Llama/Qwen API key. Model inference is a separate stage
described below. Telegram delivery isn't implemented yet.

The script runs **once**, not every ten minutes. Start with manual passes to verify
the returned media and links. Scheduling, persisted backoff, and outage alerts are
future work. Do not blindly retry a throttled or expired session every ten minutes.

## Output

```text
data/
  session.json                    # Private cookies and device/session settings
  state.json                      # Cached target ID and last fetch/success times
  stories/zero2sudo/<story-id>/
    story.json                    # Story metadata, actual links, completion flag
    media.jpg                     # Or media.mp4 for a video
```

`story.json` contains a `metadata` field with the library's normalized story object,
not the original Instagram response. A video is saved as a video; frame extraction
will be part of classification. Links are only those returned by Instagram; an
empty list does not prove the story has no clickable link. This version doesn't
open stickers in a browser, resolve redirects, or classify content.

Metadata is written before downloading. A failed download remains `complete: false`;
a later pass retries if that story is still available. Once it expires, a missing
download cannot necessarily be recovered. Finished downloads are skipped while the
story is still listed, unless their media file has gone missing or is empty.

`.env`, the virtual environment, and the default `data/` directory are ignored by
Git. Keep session files private: cookies grant account access. If you choose a custom
data directory inside the repo, add it to `.gitignore` too.

## Walkthrough of collector.py

The file has seven numbered blocks matching this explanation:

1. **Imports and configuration:** Load `.env`, validate the target handle, and resolve
   the storage directory relative to the project. This works even when a scheduler
   launches Python from another working directory.
2. **Local persistence:** Read/write JSON. Writes go to a temporary file and rename
   into place so interruptions don't leave half-written JSON. A file lock prevents
   two processes sharing the data directory from collecting concurrently.
3. **Authentication:** `--login` authenticates and saves settings. Normal passes load
   those settings and let the next Instagram request validate the session. They
   never automatically fall back to another password login.
4. **Link extraction:** Gather URLs from the model's link and sticker fields and
   remove exact duplicates. No model guesses URLs, and no external site is opened.
5. **Story storage:** Use the story ID as its folder name. Save metadata before
   downloading from the returned media URL, then mark the record complete only
   when a nonempty file exists. Existing complete records avoid repeat downloads.
6. **Collection:** Cache the creator's numeric account ID, fetch all available stories
   through the authenticated endpoint, and save each new one. Instagram client errors
   stop the run; local per-story failures leave records retryable and return failure.
   Collection timestamps distinguish a successful empty response from a failed fetch.
7. **CLI:** Parse `--login`, restrict permissions on newly created files, acquire the
   lock, create the client, and run collection. Errors return nonzero exit codes:
   `1` for local/incomplete collection, `2` for Instagram client errors. Error logs
   avoid dumping raw Instagram responses or session contents.

The small JSON state is enough for this collector. A SQLite job queue can be added
when classification and Telegram delivery introduce additional processing states.

## Offline checks

```sh
python -m unittest discover -s tests -v
```

Tests use fake stories and a fake client. They check duplicate suppression, recovery
after download failure, missing-media recovery, target ID caching, failure timestamps,
session reuse, and extraction of sticker links. They don't access Instagram.

## Local vision classifier

`classifier.py` reads the saved media. `llama.py` connects to a vision model through
an OpenAI-compatible `/v1/chat/completions` endpoint; the name doesn't require you
to use a Llama-family model. Regular Qwen3-8B is text-only; use a vision model.

For a local Ollama setup, install/start Ollama separately and download the model:

```sh
ollama pull qwen3-vl:8b
```

The Ollama application must be running (or run `ollama serve` in another terminal
if it isn't already). This project doesn't install a model or start a model server.
The download is large; make sure your current laptop has the resources for it, or
configure another vision model that fits. The planned M4 Pro/24 GB mini is the
intended host for the 8B quantized model.

Add these lines to your existing `.env`; don't overwrite your Instagram settings:

```dotenv
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=qwen3-vl:8b
LLM_API_KEY=
LLM_TIMEOUT_SECONDS=120
```

Local Ollama requires no API key. Other compatible servers can use their own URL,
exact vision-model identifier, and optional bearer key. A remote endpoint receives
the story images and extracted links. HTTP contract tests use mocked responses.
A one-story MP4 smoke test also completed against local `gemma3:4b`; this verifies
API integration and valid output, not model accuracy. Qwen inference still needs
a live run after that model is installed.

Preview inputs without calling a model, then classify a small batch:

```sh
python classifier.py --dry-run --limit 3
python classifier.py --limit 3
# Or use an already-installed local vision model without changing .env:
python classifier.py --model gemma3:4b --limit 3
```

Inspect `data/classifications/<target>/<story-id>.json` alongside each original
story image, then process the remaining stories:

```sh
python classifier.py
```

The decisions are `swe_internship`, `irrelevant`, or `uncertain`, with company/role
when visible, evidence excerpts, and a reason. There is no uncalibrated confidence
percentage. The prompt asks the model to distinguish actual software internship
openings from hiring news, advice, promotions, and full-time roles. Accuracy still
needs manual evaluation against real stories; valid JSON doesn't prove correctness.

Numbered blocks explain the code:

1. `llama.py` defines environment configuration and a strict output schema. The
   schema rejects extra fields and requires evidence for positive decisions.
2. The system prompt defines what qualifies and treats story content as untrusted
   data. Neither tools nor browsing are enabled for the model.
3. `media.py` decodes images (including HEIC), corrects orientation, and sends JPEGs
   with a maximum side of 2048 pixels as inline base64. It uses bundled FFmpeg to
   sample video at the start, middle, and 90% point; context videos use one middle
   frame. Without duration metadata it uses the first frame only. Very small text
   and brief content between samples may be missed.
4. The parser validates JSON; invalid responses remain errors, never negatives.
5. The HTTP adapter makes one bounded request without automatic retries. Server
   failures stop the batch rather than retrying all remaining stories.

`classifier.py` loads complete collector records, adds at most one story neighbor
on either side within 30 minutes, computes an input fingerprint, and writes each
result atomically. Original application URLs are copied from the target story;
the model cannot return invented URL fields. Context links aren't automatically
assigned to another story, so confirmed openings without their own links are
flagged for review. Cross-slide job/link association is future work.

Completed decisions are cached. A changed media file, prompt, model, links,
metadata, or neighboring context invalidates the result. Use `--force` to rerun
unchanged inputs. Failed responses are retryable on the next invocation. Video
results carry `media_coverage: sampled_frames_no_audio` and `review_required: true`.
A video negative is downgraded to `uncertain`, since sampled frames and no audio
cannot establish that a whole video is irrelevant. Positive video decisions retain
their visible evidence, but still warrant review. Speech transcription isn't
implemented. The classifier makes no Instagram requests and sends no Telegram messages.

Classifier tests cover the HTTP request format, output validation, caching, context
changes, retries, and video handling. These are offline software checks, not a
measurement of Qwen's accuracy.

API reference: [Ollama vision and OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility).

Library references: [story methods](https://subzeroid.github.io/instagrapi/latest/usage-guide/story/)
and [session and error handling](https://subzeroid.github.io/instagrapi/latest/usage-guide/best-practices/).
