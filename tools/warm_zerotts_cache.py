"""Warm the chi tham ZeroTTS cache on the configured Hugging Face Space.

Uses the existing 1,070 concrete TextVoice strings by default. Only Python's
standard library is needed; inference runs remotely. Successful requests are
saved locally and skipped on subsequent runs unless --rewarm is supplied.
"""

from __future__ import annotations

import hashlib
import http.client
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from tools import warm_gwen_cache as runner

VOICE_ID = "VpYLlCQEgcoHgycJstNP"
DEFAULT_URL = runner.DEFAULT_URL
DEFAULT_TEXTS = runner.DEFAULT_TEXTS
DEFAULT_OUTPUT = ROOT / "models/zerotts_cache_warmup" / VOICE_ID


def build_job(base_url: str, text: str) -> dict:
    # POST avoids URL length limits and the unified /tts provider override.
    # Omit seed so each text uses the deployed backend's default calibration.
    url = base_url.rstrip("/") + "/zerotts/synthesize"
    payload = {"voice_id": VOICE_ID, "text": text, "format": "mp3"}
    identity = json.dumps([url, payload], ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "id": hashlib.sha256(identity).hexdigest(), "text": text, "url": url,
        "method": "POST", "payload": payload, "user_agent": "zerotts-cache-warmer/1.0",
    }


def preflight(base_url: str) -> dict:
    url = base_url.rstrip("/") + "/zerotts/voices"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                data = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            detail = runner.error_detail(exc)
            if exc.code == 404:
                raise runner.WarmupError("This Space has no /zerotts/voices endpoint; deploy the ZeroTTS API first") from exc
            if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                raise runner.WarmupError(detail) from exc
        except (OSError, ValueError, http.client.HTTPException) as exc:
            detail = f"Cannot verify remote ZeroTTS support: {exc}"
            if attempt == 2:
                raise runner.WarmupError(detail) from exc
        delay = 5 * (attempt + 1)
        print(f"Waiting {delay}s for the Space: {detail}", flush=True)
        time.sleep(delay)
    voices = data.get("voices", []) if isinstance(data, dict) else []
    if not any(isinstance(voice, dict) and voice.get("voice_id") == VOICE_ID for voice in voices):
        raise runner.WarmupError(f"The remote API does not list {VOICE_ID}; no synthesis requests were sent")
    return {"checked_at": runner.timestamp(), "url": url, "status": "ok",
            "provider": "zerotts", "voice_id": VOICE_ID, "voice_count": len(voices)}


def main(argv=None) -> int:
    return runner.main(
        argv, job_builder=build_job, preflight_check=preflight,
        provider="zerotts", voice=VOICE_ID, default_output=DEFAULT_OUTPUT, description=__doc__,
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
