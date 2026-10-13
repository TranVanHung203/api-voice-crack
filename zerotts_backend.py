"""Lazy ZeroTTS backend for the FastAPI application."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import sys
import threading
import tempfile
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any


_PROJECT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA_DIR = _PROJECT_DIR / "models" / "zerotts"
_DEFAULT_BUNDLED_VOICES_DIR = _PROJECT_DIR / "voice_packs" / "zerotts"
_DEFAULT_MODEL_ID = "zeroweight-ai/ZeroTTS"
_VOICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_SAMPLING_LOCK = threading.Lock()
_SAMPLING_SEED = 21031999
_AUDIO_RAM_MAX_BYTES = 64 * 1024 * 1024


class ZeroTTSBackendError(RuntimeError):
    """Raised when ZeroTTS cannot load or synthesize audio."""


class ZeroTTSInvalidVoiceIdError(ZeroTTSBackendError):
    """Raised when a voice ID is unsafe or malformed."""


class ZeroTTSVoiceNotFoundError(ZeroTTSBackendError):
    """Raised when a requested voice pack is unavailable."""


class ZeroTTSBackend:
    """Own one lazily loaded ZeroTTS model and its local voice packs."""

    def __init__(self) -> None:
        self._engine: Any | None = None
        self._audio_lock = threading.Lock()
        self._audio_cache: OrderedDict[Path, bytes] = OrderedDict()
        self._audio_cache_bytes = 0

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    @staticmethod
    def _data_dir() -> Path:
        configured = os.getenv("ZEROTTS_DATA_DIR")
        return Path(configured).expanduser().resolve() if configured else _DEFAULT_DATA_DIR

    @classmethod
    def _voices_dir(cls) -> Path:
        return cls._data_dir() / "voices"

    @staticmethod
    def _bundled_voices_dir() -> Path:
        configured = os.getenv("ZEROTTS_BUNDLED_VOICES_DIR")
        if configured:
            return Path(configured).expanduser().resolve()
        return _DEFAULT_BUNDLED_VOICES_DIR

    @classmethod
    def _voice_roots(cls) -> tuple[tuple[str, Path], ...]:
        return (
            ("bundled", cls._bundled_voices_dir()),
            ("installed", cls._voices_dir()),
        )

    @classmethod
    def _cache_dir(cls) -> Path:
        configured = os.getenv("ZEROTTS_CACHE_DIR")
        if configured:
            return Path(configured).expanduser().resolve()
        return cls._data_dir() / "cache"

    @staticmethod
    def _validate_voice_id(voice_id: str) -> str:
        voice_id = voice_id.strip()
        if not _VOICE_ID_PATTERN.fullmatch(voice_id):
            raise ZeroTTSInvalidVoiceIdError(
                "voice_id must contain 1-128 letters, numbers, dots, hyphens, or underscores"
            )
        return voice_id

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def install_voice_pack(self, archive_path: str | Path) -> list[str]:
        """Install every ZeroTTS voice found in a downloaded ZIP archive."""
        try:
            from zerotts import install_voice_zip
        except ImportError as exc:
            raise ZeroTTSBackendError(
                f"ZeroTTS is not installed in {sys.executable}. Run: "
                f'"{sys.executable}" -m pip install zerotts==0.1.5'
            ) from exc

        voices_dir = self._voices_dir()
        voices_dir.mkdir(parents=True, exist_ok=True)
        try:
            installed = install_voice_zip(archive_path, dest_root=voices_dir)
        except Exception as exc:
            raise ZeroTTSBackendError(f"Could not install ZeroTTS voice pack: {exc}") from exc

        if self._engine is not None:
            try:
                self._engine.add_voices(voices_dir)
            except Exception as exc:
                raise ZeroTTSBackendError(
                    f"Voice pack is incompatible with the loaded ZeroTTS model: {exc}"
                ) from exc
        return sorted(installed)

    def has_local_voice(self, voice_id: str) -> bool:
        """Check a locally installed pack without loading the model."""
        try:
            voice_id = self._validate_voice_id(voice_id)
        except ZeroTTSInvalidVoiceIdError:
            return False
        return any(
            (root / voice_id / "voice.npz").is_file()
            for _source, root in self._voice_roots()
        )

    @staticmethod
    def _read_meta(voice_dir: Path) -> dict[str, Any]:
        meta_path = voice_dir / "meta.json"
        if not meta_path.is_file():
            return {}
        try:
            value = json.loads(meta_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}

    def list_voices(self) -> list[dict[str, Any]]:
        """List local packs, plus bundled voices after the model has loaded."""
        voices: dict[str, dict[str, Any]] = {}
        for source, voices_dir in self._voice_roots():
            if not voices_dir.is_dir():
                continue
            for voice_dir in sorted(voices_dir.iterdir()):
                if not voice_dir.is_dir() or not (voice_dir / "voice.npz").is_file():
                    continue
                meta = self._read_meta(voice_dir)
                voices[voice_dir.name] = {
                    "voice_id": voice_dir.name,
                    "display_name": str(meta.get("display_name", voice_dir.name)),
                    "language": str(meta.get("language", "vi")),
                    "description": str(meta.get("description", "")),
                    "tags": list(meta.get("tags", [])),
                    "source": source,
                }

        if self._engine is not None:
            model_voices_dir = Path(self._engine.model_dir) / "voices"
            for voice_id in self._engine.list_voices():
                if voice_id in voices:
                    continue
                meta = self._read_meta(model_voices_dir / voice_id)
                voices[voice_id] = {
                    "voice_id": voice_id,
                    "display_name": str(meta.get("display_name", voice_id)),
                    "language": str(meta.get("language", "vi")),
                    "description": str(meta.get("description", "")),
                    "tags": list(meta.get("tags", [])),
                    "source": "model",
                }
        return [voices[name] for name in sorted(voices)]

    def _load_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        try:
            from zerotts import ZeroTTS
        except ImportError as exc:
            raise ZeroTTSBackendError(
                f"ZeroTTS is not installed in {sys.executable}. Run: "
                f'"{sys.executable}" -m pip install zerotts==0.1.5'
            ) from exc

        configured_model_dir = os.getenv("ZEROTTS_MODEL_DIR")
        if configured_model_dir:
            model_source: str | Path = Path(configured_model_dir).expanduser().resolve()
            if not model_source.is_dir():
                raise ZeroTTSBackendError(
                    f"ZEROTTS_MODEL_DIR does not exist: {model_source}"
                )
        else:
            model_source = os.getenv("ZEROTTS_MODEL_ID", _DEFAULT_MODEL_ID)

        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        providers = [
            item.strip()
            for item in os.getenv("ZEROTTS_PROVIDERS", "CPUExecutionProvider").split(",")
            if item.strip()
        ]
        try:
            engine = ZeroTTS.from_pretrained(
                model_source,
                revision=os.getenv("ZEROTTS_REVISION") or None,
                cache_dir=str(cache_dir),
                local_files_only=self._env_flag("ZEROTTS_LOCAL_FILES_ONLY", False),
                providers=providers,
                intra_op_num_threads=int(os.getenv("ZEROTTS_THREADS", "4")),
                warmup=self._env_flag("ZEROTTS_WARMUP", True),
            )
            for _source, voices_dir in self._voice_roots():
                if voices_dir.is_dir():
                    engine.add_voices(voices_dir)
        except Exception as exc:
            raise ZeroTTSBackendError(f"Could not initialize ZeroTTS: {exc}") from exc

        self._engine = engine
        return engine

    @staticmethod
    def _encode_wav(audio: Any, sample_rate: int) -> bytes:
        import numpy as np
        import soundfile as sound_file

        output = BytesIO()
        sound_file.write(
            output,
            np.asarray(audio).squeeze(),
            sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        return output.getvalue()

    @classmethod
    def _audio_cache_dir(cls) -> Path:
        return cls._data_dir() / "audio_cache"

    def _audio_cache_key(self, text: str, voice_id: str, options: dict) -> str:
        configured_dir = os.getenv("ZEROTTS_MODEL_DIR")
        model_dirs = (
            [Path(configured_dir).expanduser().resolve()]
            if configured_dir
            else sorted(self._cache_dir().glob("models--*/snapshots/*"))
        )
        files: set[Path] = set()
        for root in model_dirs:
            files.update(root.rglob("*.onnx*"))
            files.update(root.glob("config.json"))
            files.update(root.glob("tokenizer.json"))
        voice_dirs = [root / voice_id for _, root in self._voice_roots()]
        voice_dirs.extend(root / "voices" / voice_id for root in model_dirs)
        for root in voice_dirs:
            files.update(root.glob("*"))
        fingerprints = []
        for path in sorted(files):
            try:
                if path.is_file():
                    stat = path.stat()
                    fingerprints.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
            except OSError:
                continue
        payload = {
            "version": 1, "text": text, "voice_id": voice_id,
            "seed": _SAMPLING_SEED, "options": options, "files": fingerprints,
            "model": os.getenv("ZEROTTS_MODEL_ID", _DEFAULT_MODEL_ID),
            "model_dir": configured_dir,
            "revision": os.getenv("ZEROTTS_REVISION"),
            "providers": os.getenv("ZEROTTS_PROVIDERS", "CPUExecutionProvider"),
            "threads": os.getenv("ZEROTTS_THREADS", "4"),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _remember_audio(self, path: Path, audio: bytes) -> None:
        old = self._audio_cache.pop(path, None)
        if old is not None:
            self._audio_cache_bytes -= len(old)
        if len(audio) > _AUDIO_RAM_MAX_BYTES:
            return
        self._audio_cache[path] = audio
        self._audio_cache_bytes += len(audio)
        while self._audio_cache_bytes > _AUDIO_RAM_MAX_BYTES:
            _, evicted = self._audio_cache.popitem(last=False)
            self._audio_cache_bytes -= len(evicted)

    def synthesize_wav(
        self,
        text: str,
        voice_id: str,
        *,
        cfg_scale: float = 1.0,
        audio_temperature: float = 0.8,
        audio_repetition_penalty: float = 1.2,
        max_chunk_sec: float = 15.0,
        max_frames: int = 1500,
        eoa_extra_frames: int = 1,
    ) -> bytes:
        """Reuse identical audio from RAM or disk before loading the model."""
        text = text.strip()
        voice_id = self._validate_voice_id(voice_id)
        options = {
            "cfg_scale": float(cfg_scale), "audio_temperature": float(audio_temperature),
            "audio_repetition_penalty": float(audio_repetition_penalty),
            "max_chunk_sec": float(max_chunk_sec), "max_frames": int(max_frames),
            "eoa_extra_frames": int(eoa_extra_frames),
        }
        # Also prevents duplicate work for simultaneous identical requests.
        with self._audio_lock:
            cache_dir = self._audio_cache_dir()
            path = cache_dir / f"{self._audio_cache_key(text, voice_id, options)}.wav"
            if path in self._audio_cache:
                self._audio_cache.move_to_end(path)
                return self._audio_cache[path]
            try:
                audio = path.read_bytes()
                if len(audio) > 44 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
                    self._remember_audio(path, audio)
                    return audio
            except OSError:
                pass
            audio = self._synthesize_uncached(text, voice_id, **options)
            if len(audio) <= 44:
                return audio
            # First model download can change the fingerprint during synthesis.
            path = cache_dir / f"{self._audio_cache_key(text, voice_id, options)}.wav"
            temporary = None
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".tmp", delete=False) as f:
                    temporary = Path(f.name)
                    f.write(audio)
                os.replace(temporary, path)
            except OSError as exc:
                logging.getLogger(__name__).warning("Could not cache ZeroTTS audio: %s", exc)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            self._remember_audio(path, audio)
            return audio

    def _synthesize_uncached(
        self,
        text: str,
        voice_id: str,
        *,
        cfg_scale: float = 1.0,
        audio_temperature: float = 0.8,
        audio_repetition_penalty: float = 1.2,
        max_chunk_sec: float = 15.0,
        max_frames: int = 1500,
        eoa_extra_frames: int = 1,
    ) -> bytes:
        """Synthesize one complete utterance using a ZeroTTS voice pack."""
        voice_id = self._validate_voice_id(voice_id)
        engine = self._load_engine()
        try:
            import numpy as np
            from zerotts.audio import concat_with_silence
            from zerotts.chunking import (
                chunk_text,
                clean_segment_punctuation,
                normalize_punctuation,
            )
            from zerotts.text_norm import normalize_vi_text

            # Match the web's estimated segment duration, not a hard audio cut.
            normalized = normalize_punctuation(normalize_vi_text(text))
            segments = [
                segment
                for raw in chunk_text(normalized, max_chunk_sec=max_chunk_sec)
                if (segment := clean_segment_punctuation(raw))
            ]

            # ZeroTTS 0.1.5 uses NumPy's global RNG for sampling.
            with _SAMPLING_LOCK:
                previous_state = np.random.get_state()
                try:
                    np.random.seed(_SAMPLING_SEED)
                    chunks = [
                        engine.synthesize(
                            segment,
                            voice=voice_id,
                            cfg_scale=cfg_scale,
                            audio_temperature=audio_temperature,
                            audio_repetition_penalty=audio_repetition_penalty,
                            max_frames=max_frames,
                            eoa_extra_frames=eoa_extra_frames,
                        )
                        for segment in segments
                    ]
                    audio = concat_with_silence(chunks, 0.32, engine.sample_rate)
                finally:
                    np.random.set_state(previous_state)
            return self._encode_wav(audio, engine.sample_rate)
        except FileNotFoundError as exc:
            raise ZeroTTSVoiceNotFoundError(
                f"ZeroTTS voice_id not found: {voice_id}"
            ) from exc
        except ZeroTTSBackendError:
            raise
        except Exception as exc:
            raise ZeroTTSBackendError(f"ZeroTTS could not synthesize audio: {exc}") from exc


zerotts_backend = ZeroTTSBackend()
