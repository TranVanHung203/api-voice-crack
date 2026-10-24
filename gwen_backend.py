"""Cached Gwen-TTS delivery through an isolated, persistent Python worker."""

from __future__ import annotations

import atexit
from collections import OrderedDict
import hashlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import queue
import re
import subprocess
import tempfile
import threading
import unicodedata
import uuid
import wave
from audio_cache_store import backup_audio, read_audio

ROOT = Path(__file__).resolve().parent
MODEL_ID = "g-group-ai-lab/gwen-tts-0.6B"
MODEL_REVISION = "a83e1f08656d48217a83f0ad422d1610d5b5c963"
PRESET_DIR = ROOT / "voice_packs" / "gwen"
LOGGER = logging.getLogger(__name__)


class GwenBackendError(RuntimeError):
    """The Gwen runtime could not return complete audio."""


class GwenVoiceNotFoundError(GwenBackendError):
    """Unknown Gwen speaker ID or display name."""


def load_presets() -> dict:
    try:
        return json.loads((PRESET_DIR / "data" / "ref_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GwenBackendError("Gwen voice assets missing or invalid: voice_packs/gwen/data/ref_info.json") from exc


def resolve_voice(voice_id: str | None) -> tuple[str, dict, Path]:
    def normalized(value):
        return re.sub(r"[\s_-]+", " ", unicodedata.normalize("NFC", value).strip().casefold())

    requested = normalized(voice_id or "yen_nhi")
    for key, preset in load_presets().items():
        if requested in {normalized(key), normalized(preset["name"])}:
            path = (PRESET_DIR / preset["audio_path"]).resolve()
            if not path.is_relative_to(PRESET_DIR.resolve()) or not path.is_file():
                raise GwenBackendError(f"Gwen reference audio missing: {key}")
            return key, preset, path
    raise GwenVoiceNotFoundError(f"Gwen voice not found: {voice_id}. See GET /gwen/voices")


def valid_wav(audio: bytes) -> bool:
    try:
        with wave.open(BytesIO(audio), "rb") as wav:
            return (wav.getnchannels() == 1 and wav.getsampwidth() == 2
                    and wav.getnframes() > 0
                    and len(wav.readframes(wav.getnframes())) == wav.getnframes() * 2)
    except (wave.Error, EOFError, OSError):
        return False


class GwenBackend:
    def __init__(self):
        self._lock = threading.RLock()
        self._process = None
        self._responses = None
        self._stderr = None
        self._audio_cache: OrderedDict[str, bytes] = OrderedDict()
        self._ram_limit = 32 * 1024 * 1024

    @staticmethod
    def audio_cache_dir() -> Path:
        return Path(os.getenv("GWEN_AUDIO_CACHE_DIR", str(ROOT / "models/gwen/audio_cache")))

    def list_voices(self) -> list[dict]:
        return [{"voice_id": key, "name": value["name"], "provider": "gwen",
                 "default": key == "yen_nhi"} for key, value in load_presets().items()]

    @staticmethod
    def _read_responses(stream, responses):
        try:
            for line in stream:
                responses.put(line)
        finally:
            responses.put(None)

    def _start_worker(self):
        if self._process is not None and self._process.poll() is None:
            return
        self._stop_worker()
        executable = Path(os.getenv("GWEN_PYTHON", str(
            ROOT / ".venv-gwen" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )))
        if not executable.is_file():
            raise GwenBackendError(
                "Gwen runtime missing. Run setup_gwen.ps1 (Windows) or python setup_gwen.py (Linux), "
                "or set GWEN_PYTHON to its Python executable. On Spaces, deploy setup_gwen.py and hf_app.py."
            )
        log_path = ROOT / "models/gwen/worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = log_path.open("ab")
        env = os.environ.copy()
        env.update(PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
        # Spaces enables hf_transfer globally, but it is absent in this venv.
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        env.pop("PYTHONHOME", None)
        env.setdefault("HF_HOME", str(ROOT / "models/gwen/hf-cache"))
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            self._process = subprocess.Popen(
                [str(executable), "-u", str(ROOT / "gwen_worker.py")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                text=True, encoding="utf-8", cwd=ROOT, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except OSError as exc:
            self._stop_worker()
            raise GwenBackendError(f"Could not start Gwen worker: {exc}") from exc
        self._responses = queue.Queue()
        threading.Thread(target=self._read_responses,
                         args=(self._process.stdout, self._responses), daemon=True).start()

    def _stop_worker(self):
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                if os.name == "nt":
                    # Windows venv launchers spawn a child Python: terminate only
                    # this owned worker's process tree, including its interpreter.
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, timeout=15,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if process.stdin:
                process.stdin.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def close(self):
        with self._lock:
            self._stop_worker()

    def _run_worker(self, text: str, voice_id: str, output_path: Path):
        self._start_worker()
        request_id = uuid.uuid4().hex
        try:
            message = {"id": request_id, "text": text, "voice_id": voice_id,
                       "output_path": str(output_path.resolve())}
            self._process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
            line = self._responses.get(timeout=float(os.getenv("GWEN_TIMEOUT", "900")))
            if line is None:
                self._stop_worker()
                raise GwenBackendError("Gwen worker exited. See models/gwen/worker.log")
            result = json.loads(line)
            if result.get("id") != request_id:
                self._stop_worker()
                raise GwenBackendError("Invalid Gwen worker response. See models/gwen/worker.log")
            if not result.get("ok"):
                raise GwenBackendError(f"Gwen synthesis failed: {result.get('error', 'unknown error')}")
        except queue.Empty as exc:
            self._stop_worker()
            raise GwenBackendError("Gwen inference timed out. See models/gwen/worker.log; adjust GWEN_TIMEOUT for CPU") from exc
        except (OSError, ValueError) as exc:
            self._stop_worker()
            raise GwenBackendError(f"Gwen worker communication failed: {exc}") from exc

    def _remember(self, key: str, audio: bytes) -> bytes:
        self._audio_cache.pop(key, None)
        if len(audio) <= self._ram_limit:
            self._audio_cache[key] = audio
        while sum(map(len, self._audio_cache.values())) > self._ram_limit:
            self._audio_cache.popitem(last=False)
        return audio

    def synthesize_wav(self, text: str, voice_id: str | None = None) -> bytes:
        text = unicodedata.normalize("NFC", text).strip()
        if not text:
            raise GwenBackendError("text must not be empty")
        key_voice, preset, reference = resolve_voice(voice_id)
        try:
            reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
        except OSError as exc:
            raise GwenBackendError(f"Could not read Gwen reference audio: {key_voice}") from exc
        key = hashlib.sha256(json.dumps([
            "gwen-wav-v1", text, key_voice, preset["text"],
            reference_hash,
            os.getenv("GWEN_MODEL_ID", MODEL_ID), os.getenv("GWEN_MODEL_REVISION", MODEL_REVISION),
            os.getenv("GWEN_DEVICE", "auto"), os.getenv("GWEN_DTYPE", "auto"),
        ], ensure_ascii=False).encode("utf-8")).hexdigest()
        with self._lock:
            if key in self._audio_cache:
                self._audio_cache.move_to_end(key)
                return self._audio_cache[key]
            path = self.audio_cache_dir() / f"{key}.wav"
            try:
                cached = read_audio(path, "gwen")
                if valid_wav(cached):
                    return self._remember(key, cached)
            except OSError:
                pass
            temporary = None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".wav", delete=False) as handle:
                    temporary = Path(handle.name)
                self._run_worker(text, key_voice, temporary)
                audio = temporary.read_bytes()
                if not valid_wav(audio):
                    raise GwenBackendError("Gwen worker returned empty or truncated WAV audio")
                try:
                    os.replace(temporary, path)
                except OSError:
                    LOGGER.warning("Could not persist Gwen audio cache", exc_info=True)
                backup_audio(path, "gwen", audio)
                return self._remember(key, audio)
            except OSError as exc:
                raise GwenBackendError(f"Could not access Gwen audio files: {exc}") from exc
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass


gwen_backend = GwenBackend()
atexit.register(gwen_backend.close)
