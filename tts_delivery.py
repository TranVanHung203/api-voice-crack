"""Optional MP3 delivery and HTTP validation for generated ZeroTTS audio."""

from __future__ import annotations

import hashlib
from io import BytesIO
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import Request, Response


class AudioEncodingError(RuntimeError):
    pass


_MP3_CACHE: OrderedDict[str, bytes] = OrderedDict()
_MP3_CACHE_LOCK = threading.Lock()
_MP3_ENCODE_LOCK = threading.Lock()
_MP3_RAM_LIMIT = 16 * 1024 * 1024


def _remember_mp3(key: str, audio: bytes) -> bytes:
    with _MP3_CACHE_LOCK:
        _MP3_CACHE.pop(key, None)
        if len(audio) <= _MP3_RAM_LIMIT:
            _MP3_CACHE[key] = audio
        while sum(map(len, _MP3_CACHE.values())) > _MP3_RAM_LIMIT:
            _MP3_CACHE.popitem(last=False)
    return audio


def _cached_mp3(key: str, path: Path) -> bytes | None:
    with _MP3_CACHE_LOCK:
        if key in _MP3_CACHE:
            _MP3_CACHE.move_to_end(key)
            return _MP3_CACHE[key]
    try:
        audio = path.read_bytes()
    except OSError:
        return None
    if len(audio) > 3 and (audio.startswith(b"ID3") or (audio[0] == 255 and audio[1] & 224 == 224)):
        return _remember_mp3(key, audio)
    return None


def _encode_mp3_data(wav: bytes) -> bytes:
    """Prefer the in-process codec; FFmpeg is a fallback for minimal installs."""
    try:
        import av
    except ImportError:
        av = None
    if av is not None and "libmp3lame" in av.codecs_available:
        import numpy as np
        import soundfile as sf

        try:
            samples, sample_rate = sf.read(BytesIO(wav), dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            output = BytesIO()
            with av.open(output, mode="w", format="mp3") as container:
                stream = container.add_stream("libmp3lame", rate=sample_rate)
                stream.bit_rate, stream.layout = 64000, "mono"
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples[np.newaxis, :]), format="fltp", layout="mono")
                frame.sample_rate = sample_rate
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
            return output.getvalue()
        except (av.error.FFmpegError, OSError, ValueError, RuntimeError) as exc:
            raise AudioEncodingError(f"Could not encode MP3: {type(exc).__name__}") from exc
    binary = os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not binary:
        raise AudioEncodingError("MP3 output requires av or ffmpeg; install requirements.txt or request format=wav")
    try:
        with tempfile.TemporaryDirectory(prefix="tts-mp3-") as directory:
            target = Path(directory) / "audio.mp3"
            subprocess.run(
                [binary, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "wav", "-i", "pipe:0", "-map_metadata", "-1", "-vn",
                 "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", "-threads", "1",
                 "-f", "mp3", str(target)],
                input=wav, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True, timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            return target.read_bytes()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AudioEncodingError(f"Could not encode MP3: {type(exc).__name__}") from exc


def encode_mp3(wav: bytes, cache_dir: Path) -> bytes:
    """Encode once per waveform, retaining MP3 duration/delay metadata."""
    key = hashlib.sha256(b"mp3-mono-64k-v2\0" + wav).hexdigest()
    path = cache_dir / "encoded" / f"{key}.mp3"
    cached = _cached_mp3(key, path)
    if cached is not None:
        return cached
    with _MP3_ENCODE_LOCK:
        cached = _cached_mp3(key, path)
        if cached is not None:
            return cached
        audio = _encode_mp3_data(wav)
        if not audio:
            raise AudioEncodingError("ffmpeg returned empty MP3 audio")
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                output.write(audio)
            os.replace(temporary, path)
        except OSError as exc:
            logging.getLogger(__name__).warning("Could not cache encoded audio: %s", exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return _remember_mp3(key, audio)


def audio_response(audio: bytes, media_type: str, filename: str, request: Request | None) -> Response:
    """Revalidate mutable TTS URLs; support byte ranges without altering audio."""
    etag = '"' + hashlib.sha256(audio).hexdigest() + '"'
    method = request.method if request is not None else "POST"
    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
        "Cache-Control": "private, no-cache" if method in {"GET", "HEAD"} else "no-store",
        "ETag": etag,
        "Accept-Ranges": "bytes",
    }
    if request is not None and method in {"GET", "HEAD"}:
        tags = request.headers.get("if-none-match", "")
        if any(tag.strip().removeprefix("W/") in {etag, "*"} for tag in tags.split(",")):
            return Response(status_code=304, headers=headers)
    if request is not None and method == "GET":
        value = request.headers.get("range", "")
        if_range = request.headers.get("if-range")
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
        if match and (not if_range or if_range == etag) and any(match.groups()):
            first, last = match.groups()
            if first:
                start, end = int(first), int(last) if last else len(audio) - 1
            else:
                start, end = max(0, len(audio) - int(last)), len(audio) - 1
            if (not first and int(last) == 0) or start >= len(audio) or start > end:
                headers["Content-Range"] = f"bytes */{len(audio)}"
                return Response(status_code=416, headers=headers)
            end = min(end, len(audio) - 1)
            headers["Content-Range"] = f"bytes {start}-{end}/{len(audio)}"
            return Response(audio[start:end + 1], status_code=206, media_type=media_type, headers=headers)
    if method == "HEAD":
        headers["Content-Length"] = str(len(audio))
        return Response(b"", media_type=media_type, headers=headers)
    return Response(audio, media_type=media_type, headers=headers)
