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
import wave
from collections import OrderedDict
from pathlib import Path

from fastapi import Request, Response
from audio_cache_store import backup_audio, read_audio


class AudioEncodingError(RuntimeError):
    pass


_MP3_CACHE: OrderedDict[str, bytes] = OrderedDict()
_MP3_CACHE_LOCK = threading.Lock()
_MP3_ENCODE_LOCK = threading.Lock()
_MP3_RAM_LIMIT = 16 * 1024 * 1024
_TEMPO_LOCK = threading.Lock()
_VOICE_COLOR_LOCK = threading.Lock()


def tempo_from_rate(rate: str) -> float:
    """Interpret an explicit Gwen rate; avoid extreme time stretching."""
    match = re.fullmatch(r"([+-]?\d{1,3})%", rate.strip())
    if not match or not -50 <= int(match[1]) <= 100:
        raise ValueError("Gwen rate must be a percentage from -50% to +100%, for example -20%")
    return (100 + int(match[1])) / 100


def _wav_info(audio: bytes) -> tuple[int, int, int]:
    with wave.open(BytesIO(audio), "rb") as source:
        channels, width, rate = source.getnchannels(), source.getsampwidth(), source.getframerate()
        frames = source.getnframes()
        if not frames or len(source.readframes(frames)) != frames * channels * width:
            raise ValueError("Empty or incomplete WAV")
        return channels, width, rate


def change_wav_tempo(wav: bytes, tempo: float, cache_dir: Path) -> bytes:
    """Cache an atempo derivative; retain the unmodified source WAV."""
    if not 0.5 <= tempo <= 2.0:
        raise AudioEncodingError("Gwen tempo must be between 0.5 and 2.0")
    if tempo == 1.0:
        return wav
    setting = f"{tempo:.2f}"
    key = hashlib.sha256(b"atempo-pcm16-v1\0" + setting.encode("ascii") + b"\0" + wav).hexdigest()
    path = cache_dir / "tempo" / f"{key}.wav"
    try:
        expected = _wav_info(wav)
        with _TEMPO_LOCK:
            try:
                cached = read_audio(path, "tempo")
                if _wav_info(cached) == expected:
                    return cached
            except (OSError, ValueError, wave.Error, EOFError):
                pass
            binary = os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")
            if not binary:
                raise AudioEncodingError("Changing Gwen rate requires ffmpeg; install it or omit rate")
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="tts-tempo-", dir=path.parent) as directory:
                target = Path(directory) / "audio.wav"
                subprocess.run(
                    [binary, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                     "-f", "wav", "-i", "pipe:0", "-map_metadata", "-1", "-vn",
                     "-af", f"atempo={setting}", "-c:a", "pcm_s16le", "-threads", "1",
                     "-f", "wav", str(target)],
                    input=wav, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True, timeout=120,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                audio = target.read_bytes()
                if _wav_info(audio) != expected:
                    raise AudioEncodingError("Gwen tempo processing changed the WAV audio format")
                os.replace(target, path)
                backup_audio(path, "tempo", audio)
                return audio
    except AudioEncodingError:
        raise
    except (OSError, ValueError, wave.Error, EOFError, subprocess.SubprocessError) as exc:
        raise AudioEncodingError(f"Could not change Gwen rate: {type(exc).__name__}") from exc


def change_wav_voice_color(
    wav: bytes, cache_dir: Path, *, pitch_semitones: float = 3.0, formant_ratio: float | None = None,
) -> bytes:
    """Process pitch while retaining duration; optionally shift formants independently.

    The default follows pitch with resampling/WSOLA, avoiding the spectral
    envelope estimation used by the optional Rubber Band formant path.
    """
    if (not -8 <= pitch_semitones <= 12
            or (formant_ratio is not None and not 0.8 <= formant_ratio <= 1.25)):
        raise AudioEncodingError("Voice color settings are out of range")
    pitch_semitones = round(float(pitch_semitones), 6)
    if formant_ratio is not None:
        formant_ratio = round(float(formant_ratio), 6)
    if pitch_semitones == 0 and formant_ratio in {None, 1.0}:
        return wav
    formant_setting = "follow-pitch" if formant_ratio is None else f"{formant_ratio:.6f}"
    settings = f"{pitch_semitones:.6f}:{formant_setting}".encode("ascii")
    key = hashlib.sha256(b"voice-color-pcm16-v2\0" + settings + b"\0" + wav).hexdigest()
    path = cache_dir / "voice_color" / f"{key}.wav"
    try:
        channels, width, rate = _wav_info(wav)
        if width != 2:
            raise AudioEncodingError("Voice color processing requires PCM16 WAV")
        with _VOICE_COLOR_LOCK:
            try:
                cached = read_audio(path, "voice_color")
                if _wav_info(cached) == (channels, width, rate):
                    return cached
            except (OSError, ValueError, wave.Error, EOFError):
                pass
            binary = os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")
            if not binary:
                raise AudioEncodingError("The child voice requires FFmpeg")
            pitch_ratio = 2 ** (pitch_semitones / 12)
            if formant_ratio is None:
                shifted_rate = round(rate * pitch_ratio)
                actual_pitch = shifted_rate / rate
                filters = (
                    f"asetrate={shifted_rate},aresample={rate}:filter_size=64:cutoff=0.97,"
                    f"atempo={1 / actual_pitch:.10f}"
                )
            else:
                shifted_rate = round(rate * formant_ratio)
                actual_formant = shifted_rate / rate
                filters = (
                    f"asetrate={shifted_rate},aresample={rate},"
                    f"rubberband=tempo={1 / actual_formant:.10f}:pitch={pitch_ratio / actual_formant:.10f}"
                    ":formant=preserved:pitchq=quality:transients=smooth"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="tts-voice-color-", dir=path.parent) as directory:
                target = Path(directory) / "audio.wav"
                subprocess.run(
                    [binary, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                     "-f", "wav", "-i", "pipe:0", "-map_metadata", "-1", "-vn",
                     "-af", filters, "-c:a", "pcm_f32le", "-threads", "1", "-f", "wav", str(target)],
                    input=wav, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=120,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                import numpy as np
                import soundfile as sf

                samples, output_rate = sf.read(target, dtype="float32", always_2d=True)
                if (output_rate != rate or samples.shape[1] != channels or not samples.size
                        or not np.isfinite(samples).all()):
                    raise AudioEncodingError("Invalid processed child voice audio")
                # Avoid hard clipping during PCM16 conversion after pitch shifts.
                peak = float(np.max(np.abs(samples)))
                if peak > 0.97:
                    samples *= 0.97 / peak
                output = BytesIO()
                sf.write(output, samples, rate, format="WAV", subtype="PCM_16")
                audio = output.getvalue()
                from audio_cache_store import atomic_write

                atomic_write(path, audio)
            backup_audio(path, "voice_color", audio)
            return audio
    except AudioEncodingError:
        raise
    except (OSError, ValueError, RuntimeError, wave.Error, EOFError, subprocess.SubprocessError) as exc:
        raise AudioEncodingError(
            "Could not process the child voice; check FFmpeg and the cached source WAV"
        ) from exc


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
        audio = read_audio(path, "mp3")
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
        backup_audio(path, "mp3", audio)
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
    # Inspect the complete delivered file, including MP3 gapless metadata.
    # Range responses must report the whole clip's duration, not the slice's.
    try:
        import soundfile as sf

        info = sf.info(BytesIO(audio))
        if info.frames > 0 and info.samplerate > 0:
            headers["X-Audio-Duration-Seconds"] = f"{info.frames / info.samplerate:.3f}"
    except (ImportError, OSError, ValueError, RuntimeError):
        # Metadata failure must not prevent playback of an existing cache file.
        pass
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
