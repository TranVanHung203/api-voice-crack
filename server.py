"""FastAPI server wrapping edge-tts for use with Flutter or any HTTP client."""

import asyncio
import hashlib
import os
import socket
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# To run the server, use the following command:
# uvicorn server:app --host 0.0.0.0 --port 8000
#
# LDPlayer / Android emulator: connect from inside emulator via http://10.0.2.2:8000
# Local browser / same machine:  http://127.0.0.1:8000

import edge_tts

app = FastAPI(title="edge-tts API")


@app.on_event("startup")
async def print_urls() -> None:
    port = 8000
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = "unavailable"
    print("\n" + "=" * 50)
    print("  edge-tts server is running!")
    print("=" * 50)
    print(f"  Local (same machine) : http://127.0.0.1:{port}")
    print(f"  LDPlayer / Emulator  : http://10.0.2.2:{port}")
    print(f"  LAN (other devices)  : http://{lan_ip}:{port}")
    print("=" * 50 + "\n")
    # Load persisted audio cache from disk.
    _load_disk_cache()
    # Pre-warm connection to Microsoft TTS servers by fully completing a request.
    # This puts the TCP/TLS connection back into aiohttp's pool for reuse.
    try:
        communicate = edge_tts.Communicate(text=".", voice="vi-VN-HoaiMyNeural")
        async for _ in communicate.stream():
            pass  # Consume all chunks so the connection closes cleanly and is pooled
        print("  [warmup] Connection to TTS server pre-warmed.\n")
    except Exception:
        pass  # Warmup failure is non-critical

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Audio LRU cache — persisted to disk at models/cache/<key>.mp3
# Key: MD5 of (text, voice, rate, volume, pitch).
# - Survives server restarts (loaded from disk on startup).
# - Max _CACHE_MAX_SIZE files on disk; oldest file deleted when limit exceeded.
# - Hot LRU in RAM avoids even disk reads for recently used entries.
# ---------------------------------------------------------------------------
_CACHE_DIR = Path("models/cache")
_audio_cache: OrderedDict[str, bytes] = OrderedDict()   # in-RAM hot cache
_cache_locks: dict[str, asyncio.Lock] = {}


def _make_cache_key(text: str, voice: str, rate: str, volume: str, pitch: str) -> str:
    return hashlib.md5(f"{text}|{voice}|{rate}|{volume}|{pitch}".encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.mp3"


def _load_disk_cache() -> None:
    """On startup, read existing cache files into the in-RAM OrderedDict.
    Files are ordered by modification time (oldest first) to respect LRU eviction."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(_CACHE_DIR.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
    for f in files:
        key = f.stem
        _audio_cache[key] = f.read_bytes()
    if _audio_cache:
        print(f"  [cache] Loaded {len(_audio_cache)} cached audio file(s) from disk.\n")


def _save_to_disk(key: str, data: bytes) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_bytes(data)


async def _get_audio(text: str, voice: str, rate: str, volume: str, pitch: str) -> bytes:
    """Return cached audio (RAM → disk → generate), persist new results to disk."""
    key = _make_cache_key(text, voice, rate, volume, pitch)

    # 1. RAM hit
    if key in _audio_cache:
        _audio_cache.move_to_end(key)
        return _audio_cache[key]

    # One lock per key prevents duplicate generation for concurrent requests.
    if key not in _cache_locks:
        _cache_locks[key] = asyncio.Lock()
    async with _cache_locks[key]:
        # Re-check inside the lock.
        if key in _audio_cache:
            _audio_cache.move_to_end(key)
            return _audio_cache[key]

        # 2. Disk hit (entry evicted from RAM but file still present)
        disk_file = _cache_path(key)
        if disk_file.exists():
            audio_data = disk_file.read_bytes()
            # Touch the file so its mtime reflects recent use
            os.utime(disk_file)
        else:
            # 3. Generate from Microsoft TTS
            communicate = edge_tts.Communicate(
                text=text, voice=voice, rate=rate, volume=volume, pitch=pitch
            )
            chunks: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            audio_data = b"".join(chunks)
            _save_to_disk(key, audio_data)

        _audio_cache[key] = audio_data
        _cache_locks.pop(key, None)

    return audio_data


_TTS_HEADERS = {
    "Content-Disposition": "inline; filename=tts.mp3",
    "Cache-Control": "no-store",
}


class TTSRequest(BaseModel):
    text: str
    voice: str = "vi-VN-HoaiMyNeural"
    rate: str = "-25%"
    volume: str = "+100%"
    pitch: str = "+0Hz"


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Convert text to speech. Returns cached audio instantly on repeated calls."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    audio = await _get_audio(req.text, req.voice, req.rate, req.volume, req.pitch)
    return Response(content=audio, media_type="audio/mpeg", headers=_TTS_HEADERS)


@app.get("/tts")
async def text_to_speech_get(
    text: str = Query(description="Text to synthesize"),
    voice: str = Query(default="vi-VN-HoaiMyNeural"),
    rate: str = Query(default="-10%"),
    volume: str = Query(default="+100%"),
    pitch: str = Query(default="+0Hz"),
):
    """Convert text to speech. Returns cached audio instantly on repeated calls."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    audio = await _get_audio(text, voice, rate, volume, pitch)
    return Response(content=audio, media_type="audio/mpeg", headers=_TTS_HEADERS)


@app.get("/voices")
async def list_voices(locale: str = Query(default=None, description="Filter by locale, e.g. vi-VN")):
    """List all available voices, optionally filtered by locale."""
    voices = await edge_tts.list_voices()
    if locale:
        voices = [v for v in voices if v["Locale"].lower().startswith(locale.lower())]
    return voices
