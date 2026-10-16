# Sudo scraper

Collect zero2sudo's available Instagram stories locally, for a future SWE internship
classifier and Telegram notifier. This first version only collects stories.
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

No Llama/Qwen API key is needed. Model inference and Telegram are separate stages
and aren't implemented here. A local model typically needs a server URL and model
name; whether it requires an API key depends on how that server is configured.

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

Library references: [story methods](https://subzeroid.github.io/instagrapi/latest/usage-guide/story/)
and [session and error handling](https://subzeroid.github.io/instagrapi/latest/usage-guide/best-practices/).
