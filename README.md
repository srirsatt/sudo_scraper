# zero2sudo story scraper

Collect Instagram stories from zero2sudo, save their application links, and use a
local vision model to identify internships involving coding or development.

The `swe_internship` label is intentionally inclusive: AI/ML, data science,
robotics/autonomy, tools programming, developer productivity, scripting, automation,
and any other internship with coding duties qualify. The title does not have to
say Software Engineer; a hybrid business or engineering internship with Python,
SQL, or other programming work can qualify too.

The program currently:

- Downloads available story images and videos with metadata and link-sticker URLs.
- Skips media it has already downloaded.
- Classifies saved stories as `swe_internship`, `irrelevant`, or `uncertain`.
- Optionally reads linked job descriptions first, with vision as a fallback.
- Saves decisions and supporting text for manual review.
- Sends validated matches to Telegram, with delivery tracking for each recipient.

Run each stage manually, run the full pipeline once, or keep its watcher running
to check approximately every ten minutes. Automatic startup after reboot is not
configured yet.

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
LLM_PROVIDER=ollama
LLM_MODEL=qwen3-vl:8b-instruct
LLM_CONTEXT_LENGTH=16384
LLM_API_KEY=
LLM_TIMEOUT_SECONDS=300
LLM_MAX_OUTPUT_TOKENS=1024
```

Use your own Instagram credentials, not the creator's. Your account must normally
be able to view the target's stories. Leave the password blank to enter it privately
when logging in. If you store it in `.env`, quote values containing spaces or `#`.

For local classification, install [Ollama](https://ollama.com/download), start the application, and download
the vision model:

```sh
ollama pull qwen3-vl:8b-instruct
```

If you use the CLI without the application, run `ollama serve` in another terminal.
Local Ollama needs no API key. Use **Qwen3-VL**, since regular Qwen3 is text-only.
Use the explicit `8b-instruct` tag: the plain `8b` tag can emit reasoning even when
thinking is disabled, exhausting the output budget before JSON is finished. If
your existing `.env` sets `LLM_MODEL=qwen3-vl:8b`, change it to
`LLM_MODEL=qwen3-vl:8b-instruct`. The script sends `think=false` by default; set
`OLLAMA_THINK=true` only if you intentionally want local reasoning enabled.
Other compatible vision models can be selected with `LLM_MODEL` or `--model`.
Native Ollama requests a 16K context window so story frames and nearby context fit.
For another OpenAI-compatible server, set `LLM_PROVIDER=custom`, use its `/v1` URL,
and configure the context size on that server instead. Existing configurations using
`LLM_API_STYLE` still work when `LLM_PROVIDER` isn't set.

## Use a hosted vision model

Fireworks runs classification on a hosted model instead of your laptop. Add these
settings to your existing `.env` (keep your Instagram configuration):

```dotenv
LLM_PROVIDER=fireworks
FIREWORKS_API_KEY=your_fireworks_api_key
FIREWORKS_MODEL=accounts/fireworks/models/YOUR_VISION_MODEL
```

Replace the model placeholder with a vision model or deployment path available in
your account. Choose one supporting image inputs and structured outputs; an Ollama
tag such as `qwen3-vl:8b-instruct` is not a Fireworks model ID. Catalog entries marked on-demand
require an existing deployment; this script does not deploy models. See the
[Fireworks vision documentation](https://docs.fireworks.ai/guides/querying-vision-language-models)
and [model library](https://fireworks.ai/models).

The Fireworks preset selects `https://api.fireworks.ai/inference/v1` automatically.
Your old local `LLM_BASE_URL` and `LLM_MODEL` settings do not override it. Ollama does
not need to run when using Fireworks.

```sh
python classifier.py --provider fireworks --limit 1
```

For OpenAI, set `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and `OPENAI_MODEL` to a vision
model supporting Chat Completions and structured outputs. The preset selects the
OpenAI endpoint. For other compatible endpoints, use `LLM_PROVIDER=custom`,
`LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY`. Claude's native API isn't implemented.

Hosted requests send saved story images/frames and their extracted links to the
selected provider and incur its API charges. Instagram passwords and sessions are
not sent. Keys stay in `.env` or environment variables and are never written to
classification output. There is no automatic provider fallback or request retry.
Authentication failures and throttling stop the batch and remain retryable manually.

You can switch back to local inference for a single run:

```sh
python classifier.py --provider ollama --model qwen3-vl:8b-instruct --limit 1
```

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

To try linked job text before images, use **auto mode**:

```sh
python classifier.py --provider ollama --model qwen3-vl:8b-instruct --mode auto --limit 3
```

Auto mode tries a public Greenhouse job API, `JobPosting` JSON-LD, or extracted HTML
text from a specific job-detail page. The model receives the title and description
as text, without images. Missing/blocked/ambiguous pages and ambiguous or malformed
model responses fall back to fast vision. JavaScript-only pages aren't rendered
in a browser in this version. Multiple links or multiple job listings also fall
back rather than guessing which job the story means.

Set `LLM_MODE=auto` in `.env` to make this your default. Existing fast and thorough
modes stay available; the default remains thorough until you change it.

Job pages are cached by normalized URL under `data/job_pages/`: usable extracts for
one hour, failed fetches for one minute. Existing per-story inference caching also
applies, and changed job text invalidates that decision. `--force` reruns inference
but still respects the page-fetch cache. `--dry-run` uses cached page data only and
makes no network/model requests.

Results include `classification_source` (`job_page` or `vision`), `job_page`, and
`fallback_reason` when applicable. Full extracted text is in the page-cache JSON.
Text-based decisions aren't downgraded solely because the original story was a
video. Expired structured deadlines or explicit closed-job messages set
`job_status` and require review. Missing dates or a successful page fetch do not
prove applications are still open; fetch failures are not treated as closed jobs.

Job pages are fetched separately from authenticated services: no Instagram cookies
or model API keys are sent to them. HTML/text is treated as untrusted model input.

For image-only classification, use the commands below.

Check which inputs would be processed without calling the model:

```sh
python classifier.py --dry-run --limit 4
```

Classify a small batch:

```sh
python classifier.py --limit 4
```

To try faster local classification:

```sh
python classifier.py --provider ollama --model qwen3-vl:8b-instruct --mode fast --limit 1
```

Fast mode sends one middle video frame (or the still image), resized to a maximum
side of 1024 pixels, without neighboring stories. Thorough mode uses three target
video frames and nearby context. Fast mode inspects less content, can miss small
text or cross-slide details, and still flags video results for review. Choose the
mode per run with `--mode fast` / `--mode thorough`, or set `LLM_MODE` in `.env`.
Thorough is the default. Changing modes invalidates cached decisions.

For a timing comparison, copy your current result JSONs elsewhere first, then use
`--force --limit 1` with each mode to process the same first story. Results overwrite
the previous JSON for that story. Compare `inference_seconds`, `model_metrics`, and
`model_decision`, not just the final label. Timing varies with the model, image,
generation settings, and whether the model is already loaded.

To inspect a specific saved story, replace `STORY_ID` with its folder name:

```sh
python classifier.py --provider ollama --model qwen3-vl:8b-instruct --mode fast --force --story-id STORY_ID
```

Repeat `--story-id` to select multiple stories. Thorough mode still uses neighboring
stories for context, but only the selected stories are classified. Unknown IDs fail
before any model requests are sent.

Classify all remaining uncached stories:

```sh
python classifier.py
```

To select a model for one run or reprocess unchanged inputs:

```sh
python classifier.py --model qwen3-vl:8b-instruct --limit 4
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
  job_pages/
    <url-hash>.json              # Auto-mode extracted text and fetch metadata
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
| `provider` / `model` | Which service and model produced the result. |
| `inference_seconds` | Time spent preparing media and waiting for the model. |
| `mode` / `model_metrics` | Input mode, image/token counts, and available server timings. |
| `classification_source` | Job-page text or vision used for the final model decision. |
| `job_page` / `fallback_reason` | Linked-page extraction metadata and why vision was needed. |
| `job_status` | `closed` or `expired` when explicit page evidence establishes it. |
| `decision` | The final decision after eligibility and video-review rules. |
| `policy_notes` | Why a model decision was adjusted. |
| `links` | Application links copied from the target story. |
| `review_required` | Whether manual review is needed. |
| `media_coverage` | A still image or sampled video frames without audio. |

Confirm that coding/development work and internship/co-op status are established.
A coding-focused title or quoted programming duties can establish the coding
component; the validation layer does not require Software Engineer in the title.
Pure non-coding management, advice, permanent/new-grad jobs, and roles that are
only gauging interest still do not qualify. A full-time hours label does not exclude
an internship. The classifier can still make mistakes; review both positives and
negatives before relying on its results. Prompt/policy changes invalidate old
classification caches, so the next classification pass can take longer while
applying the new definition to previously saved stories.

Thorough vision uses video frames from the start, middle, and near the end; fast
vision uses one middle frame. Audio is not transcribed, and content between sampled
frames can be missed. Vision-based video results require review; video negatives
become `uncertain` rather than being silently rejected. Job-page-based decisions
do not require review solely because the original story is a video. An empty link
list does not prove that no application link exists.

## Send matches to Telegram

1. Open the official [@BotFather](https://t.me/BotFather) in Telegram, use `/newbot`,
   and put the resulting token in your existing `.env`:

   ```dotenv
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_IDS=
   ```

2. Open your new bot and press **Start**. Friends receiving individual messages must
   also start it. For a group, add the bot; for a channel, add it as an administrator
   with permission to post. Use a dedicated bot if Hermes already manages another
   bot: chat discovery can conflict with another gateway or an active webhook.

3. Discover numeric chat IDs from recent updates:

   ```sh
   python notifier.py --list-chats
   ```

   This prints IDs and chat types, without sending messages or printing chat text.
   If the list is empty, send the bot a fresh message and try again. Discovery only
   sees recent updates, not a permanent list of everyone who started the bot.
   For a newly added channel/group, try adding the bot or posting a fresh message.
   The command does not remove webhooks or acknowledge updates with an offset.

4. Add one or more IDs to `.env`:

   ```dotenv
   TELEGRAM_CHAT_IDS=123456789,-1001234567890
   ```

Preview pending messages, then send one delivery for an initial check:

```sh
python notifier.py --dry-run
python notifier.py --limit 1
python notifier.py --status
```

Dry run makes no Telegram requests and writes no delivery state. Without a bot
token, it uses a preview identity and cannot check an actual bot's previous sends.
Set the token before relying on the preview for deduplication. You can also preview
with a recipient override: `python notifier.py --dry-run --chat-id 123456789`.

Send the remaining pending deliveries:

```sh
python notifier.py
```

Each message contains company, role, cleaned application links, and the original
story link when available. The sender reads the final `decision`, not
`model_decision`. It sends only completed `swe_internship` results with
`review_required: false`, application links, and a story timestamp within the last
24 hours. Uncertain, review-required, closed, expired, undated, and future-dated
results are held. This means a positive identified from sampled video frames is
held for review. Notification does not prove a job is still accepting applications.

`--limit` defaults to 20 **recipient deliveries**, not 20 stories; each recipient
counts separately. Repeat `--chat-id` to override the configured recipients for a
run. `--max-age-hours 48` explicitly includes older stories for testing or catch-up.
Without an override, old historical classifications are not sent.

Delivery tracking lives in `data/telegram/deliveries.sqlite3` (or under your custom
`COLLECTOR_DATA_DIR`). Keep it across runs and restarts. Deduplication is per bot,
recipient, and story, even if classification changes later; it does not deduplicate
the same job reposted in a different story. Tokens are never stored in the database.
A separate sender lock prevents simultaneous sender processes; collection can
continue independently. Do not delete delivery state unless you intend to allow
resending previously delivered stories.

| State | Behavior on the next run |
| --- | --- |
| `sent` | Skip; Telegram confirmed receipt and the message ID was saved. |
| `retry` | Retry after the saved cooldown, if the story is still eligible. |
| `rejected` | Hold; fix the token, chat, or permissions and resolve explicitly. |
| `unknown` | Hold; Telegram may have received the message. Check the chat. |
| `sending` | Hold; the process stopped before saving an outcome. Check the chat. |

Successful sends are spaced within a run. Telegram 429 responses save a bot-wide
cooldown using `retry_after` and stop the batch; connection timeouts save a one-minute
cooldown. Run the command again after the cooldown. There is no retry daemon yet.
Read timeouts, connection failures with uncertain delivery, server errors, and
malformed responses are held instead of being blindly retried. Check `--status`;
unresolved failures and deferred retries return a nonzero exit status when encountered.

After inspecting an ambiguous message in the destination chat, mark it as delivered:

```sh
python notifier.py --resolve-story STORY_ID --chat-id CHAT_ID --resolution sent
```

Or explicitly permit a retry if it was not delivered (also usable after fixing a
rejected delivery):

```sh
python notifier.py --resolve-story STORY_ID --chat-id CHAT_ID --resolution retry
python notifier.py
```

The retry still respects story age, eligibility, and any bot-wide cooldown. Telegram
`sendMessage` has no idempotency key, so exact-once delivery cannot be guaranteed
across network failures. Holding ambiguous attempts reduces accidental duplicates.
No messages are sent by `--status`, `--list-chats`, or the resolution commands.

## Run the full pipeline

After verifying Instagram login, local/hosted model access, and one Telegram
delivery, run all three stages in order:

```sh
python pipeline.py
```

The runner uses the current Python environment and the project's `.env`, even when
started from another directory. It reuses the saved Instagram session, classifier
cache, and Telegram delivery database. It does not log in again or force cached
classifications to rerun. Classification defaults to **auto mode**, overriding
`LLM_MODE` for this command; select `--mode fast` or `--mode thorough` explicitly
if desired. Provider and model still come from `.env` unless overridden.

Preview the commands without running them, making requests, or writing files:

```sh
python pipeline.py --dry-run
```

The preview shows the plan only; it does not validate credentials or preview message
contents. Use `python notifier.py --dry-run` for individual message previews.
Before a live run, local preflight checks validate configuration, the saved
Instagram session's account, and Telegram token/chat formats. They do not prove
that remote services will accept those credentials.

For continuous checks approximately every ten minutes:

```sh
python pipeline.py --watch
```

Keep the machine awake, connected, and the process running. For local inference,
Ollama must also stay running. On a Mac, you can prevent idle system sleep while
the watcher runs:

```sh
caffeinate -i python pipeline.py --watch
```

Use **Ctrl-C** to stop. This is a foreground watcher: closing its terminal,
restarting the machine, or system sleep can interrupt it. It does not install a
macOS LaunchAgent or automatically start after reboot. Ten minutes is the default
interval between pass starts; a slow pass delays the next one. There is never
more than one pass running in this watcher. A separate pipeline lock also rejects
another pipeline using the same data directory. Existing stage locks still apply
to standalone commands.

Failures stop the current pass before later stages run. In watch mode, ordinary
failures try again on the next interval using saved progress. An Instagram API
error (collector exit code 2) stops the watcher entirely; inspect your account
before manually restarting it. The runner never automatically password-logs in.
Telegram cooldowns and ambiguous-delivery holds still apply; it cannot bypass them.
A classification batch with errors holds delivery until a later successful pass.
There are no Telegram failure alerts yet; failures appear in terminal output and
the local stage records.

Useful options:

```sh
# Saved stories only; no Instagram collection request.
python pipeline.py --skip-collect

# Collect and classify without sending Telegram messages.
python pipeline.py --skip-notify

# Catch up using older stories, with one recipient delivery at most.
python pipeline.py --skip-collect --max-age-hours 72 --notify-limit 1

# Limit model work per pass or override the model/provider.
python pipeline.py --classification-limit 4 --provider ollama --model qwen3-vl:8b-instruct

# Change watch interval (minimum 60 seconds) and per-stage time budget.
python pipeline.py --watch --interval-seconds 600 --stage-timeout 3600
```

`--classification-limit` counts uncached stories; `--notify-limit` counts recipient
delivery attempts (default 20). Telegram delivery defaults to a 24-hour story age;
the runner does not restrict classification to that age. Each stage has a one-hour
timeout by default. Timed-out stages stop the pass and preserve whatever had
already been saved. Use a larger `--stage-timeout` for a large first classification
batch if needed.

Inspect the most recent executed pass:

```sh
python pipeline.py --status
```

Status is saved under `data/pipeline/last_run.json`, with each stage's outcome,
exit code, timestamps, and elapsed time. `data/pipeline/events.jsonl` appends stage
transitions across runs. Both use your custom `COLLECTOR_DATA_DIR` if set and are
ignored by Git with the default data directory. Detailed child-script output stays
in the terminal; it is not copied into the event log. Preflight failures appear in
the terminal before a pass starts and do not overwrite the last executed pass.

Pipeline exit codes: `0` successful one-shot pass, `1` failure, `2` Instagram stop,
and `130` user interruption. A hard crash can leave a `running` snapshot; the next
start warns and uses the existing caches and delivery tracking to recover. Check
`python notifier.py --status` separately for any ambiguous Telegram deliveries.

## Troubleshooting

- **Instagram login fails:** Review the sanitized `Instagram diagnostics` line and
  check the official Instagram app for verification prompts. Do not repeatedly retry
  a blocked login. For `needs_upgrade`, update dependencies and run `--login` again;
  keep `data/session.json` so the existing device identity is preserved.
- **Model unavailable:** Make sure Ollama is running, the model appears in
  `ollama list`, and `LLM_BASE_URL` includes `/v1`.
- **JSON response truncated:** Use `qwen3-vl:8b-instruct` instead of the plain `8b`
  tag, and leave `OLLAMA_THINK=false`. Error records retain output-token counts,
  reasoning-character counts, and any partial JSON; reasoning text itself isn't saved.
- **Inference times out:** Local inference now defaults to 300 seconds. Increase
  `LLM_TIMEOUT_SECONDS` in `.env` if you previously set 120, use a smaller model, or
  switch to a hosted provider. Read timeouts and connection failures have separate
  messages. Finished results remain saved; rerun without `--force` to resume.
- **Request exceeds context size:** Use `LLM_PROVIDER=ollama` and increase
  `LLM_CONTEXT_LENGTH` if needed. A larger context uses more memory.
- **Fireworks rejects the request:** Check the full model/deployment path and account
  access. HTTP 401 indicates authentication failure; HTTP 429 indicates throttling.
  Use `FIREWORKS_API_KEY`, not an OpenAI key. Hosted context limits are controlled by
  the provider, not `LLM_CONTEXT_LENGTH`. `--model` overrides need a provider-specific ID.
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
