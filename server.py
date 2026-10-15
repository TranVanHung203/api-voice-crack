"""FastAPI server wrapping edge-tts for use with Flutter or any HTTP client."""

import asyncio
import hashlib
import os
import socket
import sys
import uuid
import aiohttp
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
# To run the server, use the following command:
# uvicorn server:app --host 0.0.0.0 --port 8000
#
# LDPlayer / Android emulator: connect from inside emulator via http://10.0.2.2:8000
# Local browser / same machine:  http://127.0.0.1:8000

# Use the repository's patched Edge client for every entry point, including uvicorn.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import edge_tts
from edge_tts.exceptions import EdgeTTSException, NoAudioReceived
from vtts_backend import (
    VTTSBackendError,
    VTTSInvalidVoiceIdError,
    VTTSVoiceExistsError,
    VTTSVoiceNotFoundError,
    vtts_backend,
)
from zerotts_backend import (
    ZeroTTSBackendError,
    ZeroTTSInvalidVoiceIdError,
    ZeroTTSVoiceNotFoundError,
    zerotts_backend,
)

# Set True to override client voice IDs; set FIXED_VOICE_ID to your installed ID.
# Restart the server after changing these settings.
FORCE_VOICE_ID = False
FIXED_VOICE_ID = "hamy"


def _effective_voice_id(requested_voice_id: str | None) -> str | None:
    if not FORCE_VOICE_ID:
        return requested_voice_id
    fixed_voice_id = FIXED_VOICE_ID.strip()
    if not fixed_voice_id:
        raise HTTPException(
            status_code=503,
            detail="FIXED_VOICE_ID must be set when FORCE_VOICE_ID is enabled",
        )
    return fixed_voice_id


@asynccontextmanager
async def lifespan(_app: FastAPI):
    port = int(os.getenv("PORT", "8000"))
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
    async def warm_edge():
        communicate = edge_tts.Communicate(text=".", voice="vi-VN-HoaiMyNeural")
        async for _ in communicate.stream():
            pass  # Consume all chunks so the connection closes cleanly and is pooled
        print("  [warmup] Connection to TTS server pre-warmed.\n")
    if os.getenv("EDGE_WARMUP", "1").lower() not in {"0", "false", "no"}:
        try:
            await asyncio.wait_for(warm_edge(), timeout=5)
        except Exception:
            pass  # Warmup failure must not block startup.
    yield


app = FastAPI(title="edge-tts API vCache", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def api_home():
    return RedirectResponse("/docs")


@app.get("/health")
async def health():
    """Liveness check that does not load models or call external services."""
    return {"status": "ok"}

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
# - Hot LRU in RAM is limited to 64 MiB; disk retains generated audio.
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(os.getenv("EDGE_AUDIO_CACHE_DIR", "models/cache"))
_EDGE_RAM_MAX_BYTES = 64 * 1024 * 1024
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
    budget = _EDGE_RAM_MAX_BYTES
    selected = []
    for f in reversed(files):
        size = f.stat().st_size
        if size <= budget:
            selected.append(f)
            budget -= size
    for f in reversed(selected):
        _remember_edge_audio(f.stem, f.read_bytes())
    if _audio_cache:
        print(f"  [cache] Loaded {len(_audio_cache)} cached audio file(s) from disk.\n")


def _save_to_disk(key: str, data: bytes) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_bytes(data)


def _remember_edge_audio(key: str, data: bytes) -> None:
    _audio_cache.pop(key, None)
    if len(data) > _EDGE_RAM_MAX_BYTES:
        return
    _audio_cache[key] = data
    while sum(map(len, _audio_cache.values())) > _EDGE_RAM_MAX_BYTES:
        _audio_cache.popitem(last=False)


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
            for attempt in range(3):
                try:
                    communicate = edge_tts.Communicate(
                        text=text, voice=voice, rate=rate, volume=volume, pitch=pitch
                    )
                    chunks: list[bytes] = []
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            chunks.append(chunk["data"])
                    audio_data = b"".join(chunks)
                    if not audio_data:
                        raise NoAudioReceived("Edge TTS returned empty audio")
                    break
                except (NoAudioReceived, aiohttp.ClientConnectionError, asyncio.TimeoutError):
                    if attempt == 2:
                        raise
                    await asyncio.sleep(0.3 * (attempt + 1))
            _save_to_disk(key, audio_data)

        _remember_edge_audio(key, audio_data)
        _cache_locks.pop(key, None)

    return audio_data


_TTS_HEADERS = {
    "Content-Disposition": "inline; filename=tts.mp3",
    "Cache-Control": "no-store",
}

_VTTS_HEADERS = {
    "Content-Disposition": "inline; filename=vtts-cloned.wav",
    "Cache-Control": "no-store",
}
_ZEROTTS_HEADERS = {
    "Content-Disposition": "inline; filename=zerotts.wav",
    "Cache-Control": "no-store",
}
_VTTS_TEMP_DIR = Path(os.getenv("VTTS_TEMP_DIR", "models/vtts/tmp"))
_VTTS_MAX_REFERENCE_BYTES = int(
    os.getenv("VTTS_MAX_REFERENCE_BYTES", str(25 * 1024 * 1024))
)
_VTTS_ALLOWED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}
_vtts_inference_lock = asyncio.Lock()
_zerotts_inference_lock = asyncio.Lock()


async def _save_vtts_reference(reference_audio: UploadFile) -> Path:
    """Validate one uploaded reference clip and save it to a unique temp file."""
    suffix = Path(reference_audio.filename or "reference.wav").suffix.lower()
    if suffix not in _VTTS_ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(_VTTS_ALLOWED_SUFFIXES))
        raise HTTPException(
            status_code=400,
            detail=f"reference_audio must use one of these formats: {allowed}",
        )

    reference_data = await reference_audio.read(_VTTS_MAX_REFERENCE_BYTES + 1)
    await reference_audio.close()
    if not reference_data:
        raise HTTPException(status_code=400, detail="reference_audio must not be empty")
    if len(reference_data) > _VTTS_MAX_REFERENCE_BYTES:
        raise HTTPException(status_code=413, detail="reference_audio is too large")

    _VTTS_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    reference_path = _VTTS_TEMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    reference_path.write_bytes(reference_data)
    return reference_path


class TTSRequest(BaseModel):
    text: str
    voice: str = "vi-VN-HoaiMyNeural"
    voiceid: str | None = None
    voice_id: str | None = None
    provider: str | None = None
    eoa_extra_frames: int = Field(default=1, ge=1, le=12)
    rate: str = "-25%"
    volume: str = "+100%"
    pitch: str = "+0Hz"


class VTTSSynthesisRequest(BaseModel):
    text: str
    voice_id: str
    length_scale: float = Field(default=1.0, ge=0.25, le=4.0)
    noise_scale: float = Field(default=0.667, ge=0.0, le=2.0)
    noise_scale_w: float = Field(default=0.8, ge=0.0, le=2.0)


class ZeroTTSSynthesisRequest(BaseModel):
    text: str
    voice_id: str
    cfg_scale: float = Field(default=1.0, ge=0.0, le=5.0)
    audio_temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    audio_repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)
    max_chunk_sec: float = Field(
        default=15.0, ge=1.0, le=30.0,
        description="Estimated maximum seconds per sentence chunk; actual duration may vary",
    )
    max_frames: int = Field(default=1500, ge=4, le=1500)
    eoa_extra_frames: int = Field(default=1, ge=1, le=12)


async def _synthesize_tts_response(
    *,
    text: str,
    voice: str,
    voiceid: str | None,
    provider: str | None,
    rate: str,
    volume: str,
    pitch: str,
    eoa_extra_frames: int,
) -> Response:
    """Route a unified /tts request to Edge TTS, V-TTS, or ZeroTTS."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    voiceid = _effective_voice_id(voiceid)
    if FORCE_VOICE_ID and provider and provider.strip().lower() == "edge":
        # Edge cannot use registered IDs; preserve explicit ZeroTTS/V-TTS routing.
        provider = None
    selected_provider = provider.strip().lower() if provider else None
    if selected_provider not in {None, "edge", "vtts", "zerotts"}:
        raise HTTPException(
            status_code=400, detail="provider must be edge, vtts, or zerotts"
        )
    resolved_voiceid = voiceid.strip() if voiceid else None
    if resolved_voiceid is None and selected_provider in {"vtts", "zerotts"}:
        if voice != "vi-VN-HoaiMyNeural":
            resolved_voiceid = voice
    if (
        resolved_voiceid is None
        and selected_provider is None
        and zerotts_backend.has_local_voice(voice)
    ):
        resolved_voiceid = voice

    if selected_provider in {"vtts", "zerotts"} and not resolved_voiceid:
        raise HTTPException(
            status_code=400, detail=f"voiceid is required for provider={selected_provider}"
        )

    use_zerotts = bool(
        resolved_voiceid
        and (
            selected_provider == "zerotts"
            or (
                selected_provider is None
                and zerotts_backend.has_local_voice(resolved_voiceid)
            )
        )
    )
    if use_zerotts:
        try:
            async with _zerotts_inference_lock:
                audio = await asyncio.to_thread(
                    zerotts_backend.synthesize_wav,
                    text.strip(),
                    resolved_voiceid,
                    eoa_extra_frames=eoa_extra_frames,
                )
        except ZeroTTSVoiceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ZeroTTSInvalidVoiceIdError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ZeroTTSBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(content=audio, media_type="audio/wav", headers=_ZEROTTS_HEADERS)

    if resolved_voiceid and selected_provider != "edge":
        try:
            async with _vtts_inference_lock:
                audio = await asyncio.to_thread(
                    vtts_backend.synthesize_registered_voice,
                    text.strip(),
                    resolved_voiceid,
                )
        except VTTSVoiceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except VTTSInvalidVoiceIdError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except VTTSBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(content=audio, media_type="audio/wav", headers=_VTTS_HEADERS)

    try:
        audio = await _get_audio(text, voice, rate, volume, pitch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid Edge TTS settings: {exc}") from exc
    except (EdgeTTSException, aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Edge TTS could not generate audio: {type(exc).__name__}: {exc}",
        ) from exc
    return Response(content=audio, media_type="audio/mpeg", headers=_TTS_HEADERS)


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Unified JSON endpoint for Edge TTS, V-TTS, and ZeroTTS."""
    if not FORCE_VOICE_ID and req.voiceid and req.voice_id and req.voiceid != req.voice_id:
        raise HTTPException(status_code=400, detail="voiceid and voice_id must match")
    return await _synthesize_tts_response(
        text=req.text,
        voice=req.voice,
        voiceid=req.voiceid or req.voice_id,
        provider=req.provider,
        rate=req.rate,
        volume=req.volume,
        pitch=req.pitch,
        eoa_extra_frames=req.eoa_extra_frames,
    )


@app.get("/tts")
async def text_to_speech_get(
    text: str = Query(description="Text to synthesize"),
    voice: str = Query(default="vi-VN-HoaiMyNeural"),
    voiceid: str | None = Query(
        default=None, description="Registered V-TTS or ZeroTTS voice ID"
    ),
    provider: str | None = Query(
        default=None, description="Optional provider: edge, vtts, or zerotts"
    ),
    rate: str = Query(default="-10%"),
    volume: str = Query(default="+100%"),
    pitch: str = Query(default="+0Hz"),
    eoa_extra_frames: int = Query(default=1, ge=1, le=12),
):
    """Unified query endpoint for Edge TTS, V-TTS, and ZeroTTS."""
    return await _synthesize_tts_response(
        text=text,
        voice=voice,
        voiceid=voiceid,
        provider=provider,
        rate=rate,
        volume=volume,
        pitch=pitch,
        eoa_extra_frames=eoa_extra_frames,
    )


@app.post("/vtts/clone")
async def clone_voice_vtts(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    length_scale: float = Form(default=1.0),
    noise_scale: float = Form(default=0.667),
    noise_scale_w: float = Form(default=0.8),
):
    """Clone a voice from a short reference clip and return one complete WAV."""
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if not 0.25 <= length_scale <= 4.0:
        raise HTTPException(
            status_code=400, detail="length_scale must be between 0.25 and 4.0"
        )
    if not 0.0 <= noise_scale <= 2.0 or not 0.0 <= noise_scale_w <= 2.0:
        raise HTTPException(
            status_code=400,
            detail="noise_scale and noise_scale_w must be between 0.0 and 2.0",
        )

    reference_path = await _save_vtts_reference(reference_audio)
    try:
        async with _vtts_inference_lock:
            audio = await asyncio.to_thread(
                vtts_backend.synthesize_wav,
                text,
                reference_path,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
            )
    except VTTSBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        reference_path.unlink(missing_ok=True)

    return Response(content=audio, media_type="audio/wav", headers=_VTTS_HEADERS)


@app.post("/vtts/voices", status_code=201)
async def register_vtts_voice(
    reference_audio: UploadFile = File(...),
    voice_id: str | None = Form(default=None),
):
    """Register a reference clip once and persist its voice embeddings."""
    reference_path = await _save_vtts_reference(reference_audio)
    try:
        async with _vtts_inference_lock:
            registered_id = await asyncio.to_thread(
                vtts_backend.register_voice, reference_path, voice_id
            )
    except VTTSVoiceExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except VTTSInvalidVoiceIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VTTSBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        reference_path.unlink(missing_ok=True)
    return {"voice_id": registered_id}


@app.get("/vtts/voices")
async def list_vtts_voices():
    """List voice profiles registered on this server."""
    return {"voices": vtts_backend.list_voices()}


@app.post("/vtts/synthesize")
async def synthesize_registered_vtts_voice(req: VTTSSynthesisRequest):
    """Synthesize a complete WAV using text and a registered voice ID."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    voice_id = _effective_voice_id(req.voice_id)
    try:
        async with _vtts_inference_lock:
            audio = await asyncio.to_thread(
                vtts_backend.synthesize_registered_voice,
                text,
                voice_id,
                length_scale=req.length_scale,
                noise_scale=req.noise_scale,
                noise_scale_w=req.noise_scale_w,
            )
    except VTTSVoiceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except VTTSInvalidVoiceIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VTTSBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(content=audio, media_type="audio/wav", headers=_VTTS_HEADERS)


@app.get("/zerotts/voices")
async def list_zerotts_voices():
    """List installed ZeroTTS packs without forcing a model download."""
    return {"voices": zerotts_backend.list_voices()}


@app.post("/zerotts/synthesize")
async def synthesize_zerotts_voice(req: ZeroTTSSynthesisRequest):
    """Synthesize a complete PCM WAV using an installed ZeroTTS voice pack."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    voice_id = _effective_voice_id(req.voice_id)
    try:
        async with _zerotts_inference_lock:
            audio = await asyncio.to_thread(
                zerotts_backend.synthesize_wav,
                text,
                voice_id,
                cfg_scale=req.cfg_scale,
                audio_temperature=req.audio_temperature,
                audio_repetition_penalty=req.audio_repetition_penalty,
                max_chunk_sec=req.max_chunk_sec,
                max_frames=req.max_frames,
                eoa_extra_frames=req.eoa_extra_frames,
            )
    except ZeroTTSVoiceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ZeroTTSInvalidVoiceIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ZeroTTSBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(content=audio, media_type="audio/wav", headers=_ZEROTTS_HEADERS)


@app.get("/voices")
async def list_voices(locale: str = Query(default=None, description="Filter by locale, e.g. vi-VN")):
    """List all available voices, optionally filtered by locale."""
    voices = await edge_tts.list_voices()
    if locale:
        voices = [v for v in voices if v["Locale"].lower().startswith(locale.lower())]
    return voices
