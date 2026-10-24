"""Sequentially warm remote Gwen/Yen Nhi caches using only a UTF-8 JSON text list."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://tranvanhung203-edge-tts-api.hf.space"
DEFAULT_TEXTS = ROOT / "data/gwen_cache/texts.json"


class WarmupError(RuntimeError):
    pass


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")


def load_texts(path: Path) -> list[str]:
    values = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(values, list) or not all(isinstance(text, str) for text in values):
        raise WarmupError("The text file must be a JSON array of strings")
    texts = list(dict.fromkeys(text.strip() for text in values if text.strip()))
    invalid = [text for text in texts if re.search(r"\$\{|\$[A-Za-z_]", text)]
    if invalid:
        raise WarmupError(f"The text file still contains {len(invalid)} unresolved Dart templates")
    if not texts:
        raise WarmupError("The text file is empty")
    return texts


def build_job(base_url: str, text: str) -> dict:
    url = base_url.rstrip("/") + "/tts?" + urllib.parse.urlencode({
        "provider": "gwen", "voice": "yen_nhi", "text": text, "format": "mp3",
    })
    return {"id": hashlib.sha256(url.encode("utf-8")).hexdigest(), "text": text, "url": url}


def error_detail(error) -> str:
    if isinstance(error, urllib.error.HTTPError):
        body = error.read(8192).decode("utf-8", errors="replace")
        try:
            data = json.loads(body)
            detail = data.get("detail", data.get("error", body))
        except (ValueError, AttributeError):
            detail = body
        return f"HTTP {error.code}: {str(detail)[:600]}"
    return f"{type(error).__name__}: {error}"


def preflight(base_url: str) -> dict:
    url = base_url.rstrip("/") + "/gwen/voices"
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise WarmupError(
                "This Space has no /gwen/voices endpoint. Deploy the Gwen adapter, "
                "reference files and isolated runtime before warming its cache. "
                "No /tts requests were sent."
            ) from exc
        raise WarmupError(error_detail(exc)) from exc
    except (OSError, ValueError) as exc:
        raise WarmupError(f"Cannot verify remote Gwen support: {exc}") from exc
    voices = data.get("voices", []) if isinstance(data, dict) else []
    if not any(isinstance(voice, dict) and voice.get("voice_id") == "yen_nhi" for voice in voices):
        raise WarmupError("The remote API does not list yen_nhi; no /tts requests were sent")
    return {"checked_at": timestamp(), "url": url, "status": "ok", "voice_count": len(voices)}


def previous_successes(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue  # A process interrupted during the final append can leave a partial line.
            if record.get("status") == "ok":
                done.add(record["id"])
            elif record.get("status") == "error":
                done.discard(record["id"])
            elif record.get("status") == "reset":
                done.clear()
    return done


def append_receipt(path: Path, record: dict) -> None:
    needs_newline = False
    if path.exists() and path.stat().st_size:
        with path.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            needs_newline = existing.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as output:
        if needs_newline:
            output.write("\n")
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def fetch_audio(job: dict, output_dir: Path, timeout: float, attempts: int) -> dict:
    last_error = None
    for attempt in range(1, attempts + 1):
        temporary = None
        retry = True
        retry_after = 0
        try:
            payload = job.get("payload")
            request = urllib.request.Request(
                job["url"],
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None,
                method=job.get("method", "GET"),
                headers={
                    "Accept": "audio/mpeg", "Content-Type": "application/json",
                    "User-Agent": job.get("user_agent", "gwen-cache-warmer/1.0"),
                },
            )
            start = time.monotonic()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise WarmupError(f"Expected full audio (200), got {response.status}")
                content_type = response.headers.get("Content-Type", "").split(";")[0]
                if content_type not in {"audio/mpeg", "audio/mp3"}:
                    raise WarmupError(f"Expected MP3 audio, got {content_type!r}")
                size, first = 0, b""
                digest = hashlib.sha256()
                with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".part", delete=False) as out:
                    temporary = Path(out.name)
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        if not first:
                            first = chunk[:16]
                        digest.update(chunk)
                        size += len(chunk)
                        out.write(chunk)
                length = response.headers.get("Content-Length")
                if length is not None and size != int(length):
                    raise WarmupError(f"Incomplete audio: expected {length} bytes, got {size}")
                if size < 4 or not (first.startswith(b"ID3") or (first[0] == 255 and first[1] & 224 == 224)):
                    raise WarmupError("The API returned empty or invalid MP3 data")
                destination = output_dir / (job["id"] + ".mp3")
                os.replace(temporary, destination)
                return {"status": "ok", "bytes": size, "sha256": digest.hexdigest(),
                        "etag": response.headers.get("ETag"), "audio_file": str(destination.resolve()),
                        "seconds": round(time.monotonic() - start, 2), "attempt": attempt}
        except urllib.error.HTTPError as exc:
            retry = exc.code in {408, 429, 500, 502, 503, 504}
            retry_after = int(exc.headers.get("Retry-After", "0")) if exc.headers.get("Retry-After", "0").isdigit() else 0
            last_error = error_detail(exc)
            if any(marker in last_error for marker in ["Gwen runtime missing", "setup_gwen.ps1", "GWEN_PYTHON"]):
                retry = False
        except (OSError, ValueError, WarmupError, http.client.HTTPException) as exc:
            last_error = error_detail(exc)
            if isinstance(exc, WarmupError):
                retry = False
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if not retry or attempt == attempts:
            break
        delay = max(retry_after, min(60, 5 * 2 ** (attempt - 1)))
        print(f"  retry {attempt}/{attempts - 1} in {delay}s: {last_error}", flush=True)
        time.sleep(delay)
    raise WarmupError(last_error or "Audio request failed")


def main(argv=None, *, job_builder=None, preflight_check=None,
         provider="gwen", voice="yen_nhi", default_output=None, description=None) -> int:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--texts", type=Path, default=DEFAULT_TEXTS)
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=default_output or ROOT / "models/gwen_cache_warmup")
    parser.add_argument("--limit", type=int, default=0, help="Maximum pending texts this run; 0 = all")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.5, help="Pause between sequential requests")
    parser.add_argument("--rewarm", action="store_true", help="Request again even if a previous run succeeded")
    parser.add_argument("--dry-run", action="store_true", help="Write request list without network calls")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.timeout <= 0 or args.attempts < 1 or args.delay < 0:
        parser.error("limit/delay must be nonnegative; timeout/attempts must be positive")
    if urllib.parse.urlsplit(args.base_url).scheme not in {"http", "https"}:
        parser.error("base-url must be an HTTP(S) URL")
    try:
        texts = load_texts(args.texts)
        args.output.mkdir(parents=True, exist_ok=True)
        journal = args.output / "progress.jsonl"
        completed = set() if args.rewarm else previous_successes(journal)
        builder = job_builder or build_job
        jobs = [builder(args.base_url, text) for text in texts]
        pending = [job for job in jobs if job["id"] not in completed]
        skipped = len(jobs) - len(pending)
        if args.limit:
            pending = pending[:args.limit]
        requests_file = args.output / "requests.jsonl"
        requests_file.write_text("".join(json.dumps(job, ensure_ascii=False) + "\n" for job in pending), encoding="utf-8")
        print(f"Texts: {len(texts)} | previously successful: {skipped} | selected: {len(pending)}", flush=True)
        print(f"Provider: {provider} | voice: {voice} | API: {args.base_url}", flush=True)
        if args.dry_run or not pending:
            print(f"{'Dry run' if args.dry_run else 'Nothing pending'}: {requests_file}")
            return 0
        try:
            capability = (preflight_check or preflight)(args.base_url)
        except WarmupError as exc:
            (args.output / "preflight.json").write_text(json.dumps({
                "checked_at": timestamp(), "base_url": args.base_url, "status": "blocked", "error": str(exc),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise
        (args.output / "preflight.json").write_text(json.dumps(capability, indent=2) + "\n", encoding="utf-8")
        if args.rewarm:
            # A new remote cache population must not inherit successes from
            # before a Space restart. An interrupted rewarm can then resume
            # normally, without repeating the already re-created entries.
            append_receipt(journal, {"status": "reset", "time": timestamp(), "base_url": args.base_url})
        audio_dir = args.output / "audio"
        audio_dir.mkdir(exist_ok=True)
        success = failed = consecutive_failures = 0
        for index, job in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] {job['text'].replace(chr(10), ' ')[:100]}", flush=True)
            try:
                result = fetch_audio(job, audio_dir, args.timeout, args.attempts)
                success += 1
                consecutive_failures = 0
                print(f"  OK {result['bytes']} bytes, {result['seconds']}s", flush=True)
            except WarmupError as exc:
                result = {"status": "error", "error": str(exc)}
                failed += 1
                consecutive_failures += 1
                print(f"  ERROR {exc}", flush=True)
            append_receipt(journal, {"time": timestamp(), **job, **result})
            if consecutive_failures >= 3:
                print("Stopped after 3 consecutive failures; fix the API and rerun to resume.")
                break
            if args.delay:
                time.sleep(args.delay)
        print(f"Finished: {success} successful, {failed} failed. Progress: {journal}")
        return 1 if failed else 0
    except (WarmupError, OSError, ValueError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Successful requests are saved; rerun to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
