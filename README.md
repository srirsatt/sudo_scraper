# zero2sudo story scraper

Collect Instagram stories from zero2sudo, save their application links, and use a
local vision model to identify software engineering internship postings.

The program currently:

- Downloads available story images and videos with metadata and link-sticker URLs.
- Skips media it has already downloaded.
- Classifies saved stories as `swe_internship`, `irrelevant`, or `uncertain`.
- Saves decisions and supporting text for manual review.

Collection and classification run manually, once per command. Automatic scheduling
and Telegram notifications are not implemented yet.

## Setup

Use Python 3.10+ on macOS or Linux. Run these commands from the project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a fresh setup, create your local configuration:

```sh
cp .env.example .env
```

If you already have a `.env`, keep it and add any missing settings from
[.env.example](.env.example). Edit the settings for your account:

```dotenv
INSTAGRAM_USERNAME=your_instagram_username
INSTAGRAM_PASSWORD=
INSTAGRAM_TARGET=zero2sudo
COLLECTOR_DATA_DIR=data

LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=qwen3-vl:8b
LLM_API_KEY=
LLM_TIMEOUT_SECONDS=120
```

Use your own Instagram credentials, not the creator's. Your account must normally
be able to view the target's stories. Leave the password blank to enter it privately
when logging in. If you store it in `.env`, quote values containing spaces or `#`.

Install [Ollama](https://ollama.com/download), start the application, and download
the vision model:

```sh
ollama pull qwen3-vl:8b
```

If you use the CLI without the application, run `ollama serve` in another terminal.
Local Ollama needs no API key. Use **Qwen3-VL**, since regular Qwen3 is text-only.
Other compatible vision models can be selected with `LLM_MODEL` or `--model`.

## Collect stories

Log in and collect the currently available stories:

```sh
python collector.py --login
```

Enter your password and two-factor code if prompted. The collector saves the
session locally. For subsequent collection passes, reuse it:

```sh
python collector.py
```

The summary reports how many stories were available, newly saved, or incomplete.
If nothing new has appeared, `newly saved: 0` is expected. Failed downloads can retry
on a later pass while the story is still available.

## Classify saved stories

Check which inputs would be processed without calling the model:

```sh
python classifier.py --dry-run --limit 4
```

Classify a small batch:

```sh
python classifier.py --limit 4
```

Classify all remaining uncached stories:

```sh
python classifier.py
```

To select a model for one run or reprocess unchanged inputs:

```sh
python classifier.py --model qwen3-vl:8b --limit 4
python classifier.py --force --limit 4
```

`--limit` counts uncached stories, so repeating a limited run processes the next
pending batch. Changing the model, prompt, processing rules, or story inputs also
invalidates the cache. Results for a story are overwritten when it is reprocessed;
copy them elsewhere first if you want to retain a model comparison.

Classification reads saved media without contacting Instagram. Original application
URLs come from the collector; the model does not supply them.

## Inspect results

With the default configuration, files are stored here:

```text
data/
  session.json
  state.json
  stories/zero2sudo/<story-id>/
    story.json
    media.heic                  # Filename/extension varies; videos use .mp4
  classifications/zero2sudo/
    <story-id>.json
```

Open a classification JSON and compare it with the media in the matching story-ID
folder. To open the result folder in VS Code, if its `code` command is installed:

```sh
code data/classifications/zero2sudo
```

The main result fields are:

| Field | Meaning |
| --- | --- |
| `status` | `complete` for processed results; `error` remains retryable. |
| `model_decision` | The model's original label, company, role, evidence, and reason. |
| `decision` | The final decision after eligibility and video-review rules. |
| `policy_notes` | Why a model decision was adjusted. |
| `links` | Application links copied from the target story. |
| `review_required` | Whether manual review is needed. |
| `media_coverage` | A still image or sampled video frames without audio. |

Confirm that an actual software engineering/development role and internship/co-op
status are visible. A tech-company job, product-management internship, or Engineer I
title alone is not enough. The classifier can still make mistakes; review both
positives and negatives before relying on its results.

Videos use sampled frames from the start, middle, and near the end. Audio is not
transcribed, and content between sampled frames can be missed. Every video result
requires review; video negatives become `uncertain` rather than being silently
rejected. An empty link list does not prove that no application link exists.

## Troubleshooting

- **Instagram login fails:** Review the sanitized `Instagram diagnostics` line and
  check the official Instagram app for verification prompts. Do not repeatedly retry
  a blocked login. For `needs_upgrade`, update dependencies and run `--login` again;
  keep `data/session.json` so the existing device identity is preserved.
- **Model unavailable:** Make sure Ollama is running, the model appears in
  `ollama list`, and `LLM_BASE_URL` includes `/v1`.
- **Inference times out:** Increase `LLM_TIMEOUT_SECONDS` in `.env` or try a smaller
  vision model. Failed classifications remain retryable.
- **No stories found:** Check `INSTAGRAM_TARGET`, `COLLECTOR_DATA_DIR`, and whether
  collection completed successfully.

`.env` and the default `data/` directory are Git-ignored. Keep passwords and session
cookies private. Add any custom data directory inside the repository to `.gitignore`.
Instagram access uses an unofficial client and can be restricted; complete coverage
and a fixed detection interval are not guaranteed.

## Run offline checks

```sh
python -m unittest discover -s tests -v
```

These tests check the software without Instagram or model requests. They do not
measure the model's classification accuracy.
