"""Lazy, serialized VieNeu-TTS v3 Turbo synthesis for the unified API."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from audio_cache_store import backup_audio, read_audio


class VieNeuBackendError(RuntimeError):
    """The optional runtime or model could not generate audio."""


class VieNeuVoiceNotFoundError(VieNeuBackendError):
    """The requested preset voice does not exist."""


class VieNeuBackend:
    def __init__(self):
        self._engine = None
        self._audio_cache: OrderedDict[str, bytes] = OrderedDict()
        self._ram_limit = 32 * 1024 * 1024
        # Keep this lock in the worker thread: cancelling an HTTP request must
        # not allow another inference to mutate the model while it is running.
        self._lock = threading.RLock()

    def _load_engine(self):
        if self._engine is None:
            try:
                from vieneu import Vieneu

                self._engine = Vieneu(
                    mode="v3turbo",
                    backend=os.getenv("VIENEU_BACKEND", "onnx"),
                    device=os.getenv("VIENEU_DEVICE", "cpu"),
                    precision="fp32",
                    threads=int(os.getenv("VIENEU_THREADS", "4")),
                )
            except ImportError as exc:
                raise VieNeuBackendError(
                    "VieNeu runtime missing; install requirements.txt with the API's Python environment"
                ) from exc
            except Exception as exc:
                raise VieNeuBackendError(f"Could not load VieNeu-TTS: {exc}") from exc
        return self._engine

    @staticmethod
    def audio_cache_dir() -> Path:
        return Path(os.getenv("VIENEU_AUDIO_CACHE_DIR", "models/vieneu/audio_cache"))

    def list_voices(self) -> list[dict]:
        with self._lock:
            engine = self._load_engine()
            return [
                {"voice_id": voice_id, "label": label, "provider": "vieneu"}
                for label, voice_id in engine.list_preset_voices()
            ]

    def _remember(self, key: str, audio: bytes) -> bytes:
        self._audio_cache.pop(key, None)
        if len(audio) <= self._ram_limit:
            self._audio_cache[key] = audio
        while sum(map(len, self._audio_cache.values())) > self._ram_limit:
            self._audio_cache.popitem(last=False)
        return audio

    def synthesize_wav(self, text: str, voice_id: str | None = None) -> bytes:
        with self._lock:
            key = hashlib.sha256(json.dumps([
                "vieneu-3.8.3-wav-v1", os.getenv("VIENEU_BACKEND", "onnx"),
                os.getenv("VIENEU_DEVICE", "cpu"), text, voice_id,
            ], ensure_ascii=False).encode("utf-8")).hexdigest()
            if key in self._audio_cache:
                self._audio_cache.move_to_end(key)
                return self._audio_cache[key]
            path = self.audio_cache_dir() / f"{key}.wav"
            try:
                cached = read_audio(path, "vieneu")
                if len(cached) > 44 and cached[:4] == b"RIFF" and cached[8:12] == b"WAVE":
                    return self._remember(key, cached)
            except OSError:
                pass
            engine = self._load_engine()
            # The upstream SDK can fall back to its default for unknown names;
            # an API client should get an actionable error instead.
            if voice_id and engine.resolve_voice_name(voice_id) is None:
                raise VieNeuVoiceNotFoundError(
                    f"VieNeu voice not found: {voice_id}. See GET /vieneu/voices"
                )
            try:
                import numpy as np
                import soundfile as sf

                audio = np.asarray(engine.infer(text, voice=voice_id), dtype=np.float32)
                if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
                    raise ValueError("model returned empty or invalid audio")
                output = BytesIO()
                sf.write(output, audio, engine.sample_rate, format="WAV", subtype="PCM_16")
                encoded = output.getvalue()
            except Exception as exc:
                raise VieNeuBackendError(f"VieNeu synthesis failed: {exc}") from exc
            temporary = None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(encoded)
                os.replace(temporary, path)
            except OSError:
                logging.getLogger(__name__).warning("Could not persist VieNeu audio cache", exc_info=True)
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
            backup_audio(path, "vieneu", encoded)
            return self._remember(key, encoded)


vieneu_backend = VieNeuBackend()
