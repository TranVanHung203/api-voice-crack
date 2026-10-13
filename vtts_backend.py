"""Lazy V-TTS zero-shot backend used by the FastAPI application."""

from __future__ import annotations

import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any


_PROJECT_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_DIR = _PROJECT_DIR / "models" / "vtts"
_DEFAULT_VTTS_SPACE = "letrggghieu/v-zeroshot-voice-cloning"
_VOICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class VTTSBackendError(RuntimeError):
    """Raised when V-TTS cannot be loaded or cannot synthesize audio."""


class VTTSVoiceExistsError(VTTSBackendError):
    """Raised when a requested voice ID already exists."""


class VTTSInvalidVoiceIdError(VTTSBackendError):
    """Raised when a voice ID is not safe to persist as a filename."""


class VTTSVoiceNotFoundError(VTTSBackendError):
    """Raised when a requested voice ID does not exist."""


class VTTSBackend:
    """Own one lazily loaded V-TTS model instance."""

    def __init__(self) -> None:
        self._engine: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    @staticmethod
    def _model_dir() -> Path:
        configured = os.getenv("VTTS_MODEL_DIR")
        return Path(configured).expanduser().resolve() if configured else _DEFAULT_MODEL_DIR

    @classmethod
    def _voices_dir(cls) -> Path:
        return cls._model_dir() / "voices"

    @staticmethod
    def _validate_voice_id(voice_id: str) -> str:
        voice_id = voice_id.strip()
        if not _VOICE_ID_PATTERN.fullmatch(voice_id):
            raise VTTSInvalidVoiceIdError(
                "voice_id must contain 1-64 letters, numbers, hyphens, or underscores"
            )
        return voice_id

    @staticmethod
    def _download_asset(filename: str, model_dir: Path) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise VTTSBackendError(
                "Thiếu huggingface-hub. Hãy cài lại dependencies từ requirements.txt."
            ) from exc

        try:
            downloaded = hf_hub_download(
                repo_id=os.getenv("VTTS_HF_REPO", _DEFAULT_VTTS_SPACE),
                repo_type="space",
                filename=filename,
                local_dir=str(model_dir),
            )
        except Exception as exc:
            raise VTTSBackendError(f"Không tải được model V-TTS: {exc}") from exc
        return Path(downloaded)

    def _resolve_assets(self) -> tuple[Path, Path, Path]:
        model_dir = self._model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_value = os.getenv("VTTS_CHECKPOINT_PATH")
        config_value = os.getenv("VTTS_CONFIG_PATH")
        hasp_value = os.getenv("VTTS_HASP_PATH")

        checkpoint = (
            Path(checkpoint_value).expanduser().resolve()
            if checkpoint_value
            else model_dir / "pretrained" / "zeroshot" / "G_175000.pth"
        )
        config = (
            Path(config_value).expanduser().resolve()
            if config_value
            else model_dir / "pretrained" / "zeroshot" / "config.json"
        )
        hasp = (
            Path(hasp_value).expanduser().resolve()
            if hasp_value
            else model_dir / "pretrained" / "hasp" / "pytorch_model.bin"
        )

        if not checkpoint.is_file():
            if checkpoint_value:
                raise VTTSBackendError(f"Không tìm thấy checkpoint V-TTS: {checkpoint}")
            checkpoint = self._download_asset(
                "pretrained/zeroshot/G_175000.pth", model_dir
            )
        if not config.is_file():
            if config_value:
                raise VTTSBackendError(f"Không tìm thấy config V-TTS: {config}")
            config = self._download_asset("pretrained/zeroshot/config.json", model_dir)
        if not hasp.is_file():
            if hasp_value:
                raise VTTSBackendError(f"Không tìm thấy H/ASP model: {hasp}")
            hasp = self._download_asset(
                "pretrained/hasp/pytorch_model.bin", model_dir
            )

        return checkpoint, config, hasp

    @staticmethod
    def _load_hasp_weights(encoder: Any, hasp_path: Path) -> None:
        """Load H/ASP from our model directory instead of V-TTS's cwd cache."""
        import torch

        checkpoint = torch.load(str(hasp_path), map_location="cpu", weights_only=False)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        state_dict = {
            key.replace("module.", ""): value for key, value in state_dict.items()
        }
        encoder.model.load_state_dict(state_dict, strict=False)

    def _load_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        try:
            from v_tts import ZeroShotTTS
            from src.models.encoders import SpeakerEncoder
        except ImportError as exc:
            raise VTTSBackendError(
                "V-TTS chưa được cài. Hãy chạy: pip install -r requirements.txt"
            ) from exc

        checkpoint, config, hasp = self._resolve_assets()
        device = os.getenv("VTTS_DEVICE", "cpu")

        # V-TTS normally looks for H/ASP under the current working directory.
        # Patch only while constructing the engine, then restore the class method.
        original_loader = SpeakerEncoder._load_pretrained

        def load_configured_hasp(encoder: Any) -> None:
            self._load_hasp_weights(encoder, hasp)

        SpeakerEncoder._load_pretrained = load_configured_hasp
        try:
            self._engine = ZeroShotTTS(
                checkpoint_path=str(checkpoint),
                config_path=str(config),
                device=device,
            )
        except Exception as exc:
            raise VTTSBackendError(f"Không khởi tạo được V-TTS: {exc}") from exc
        finally:
            SpeakerEncoder._load_pretrained = original_loader

        return self._engine

    def register_voice(self, reference_audio: Path, voice_id: str | None = None) -> str:
        """Extract and persist voice embeddings from one reference recording."""
        import numpy as np

        voice_id = self._validate_voice_id(voice_id or uuid.uuid4().hex)
        voices_dir = self._voices_dir()
        voices_dir.mkdir(parents=True, exist_ok=True)
        voice_path = voices_dir / f"{voice_id}.npz"
        if voice_path.exists():
            raise VTTSVoiceExistsError(f"voice_id already exists: {voice_id}")

        engine = self._load_engine()
        temporary_path = voices_dir / f".{voice_id}.{uuid.uuid4().hex}.tmp"
        try:
            speaker, prosody = engine.extract_embeddings(str(reference_audio))
            with temporary_path.open("wb") as output:
                np.savez_compressed(
                    output,
                    speaker=speaker.detach().cpu().numpy(),
                    prosody=prosody.detach().cpu().numpy(),
                )
            temporary_path.replace(voice_path)
        except VTTSBackendError:
            raise
        except Exception as exc:
            raise VTTSBackendError(f"Không tạo được voice profile: {exc}") from exc
        finally:
            temporary_path.unlink(missing_ok=True)
        return voice_id

    def list_voices(self) -> list[dict[str, int | str]]:
        """List persisted voice profiles without loading the model."""
        voices_dir = self._voices_dir()
        if not voices_dir.exists():
            return []
        voices: list[dict[str, int | str]] = []
        for voice_path in sorted(voices_dir.glob("*.npz")):
            stat = voice_path.stat()
            voices.append(
                {
                    "voice_id": voice_path.stem,
                    "size_bytes": stat.st_size,
                    "modified_at": int(stat.st_mtime),
                }
            )
        return voices

    def _load_voice_embeddings(self, voice_id: str) -> tuple[Any, Any]:
        import numpy as np
        import torch

        voice_id = self._validate_voice_id(voice_id)
        voice_path = self._voices_dir() / f"{voice_id}.npz"
        if not voice_path.is_file():
            raise VTTSVoiceNotFoundError(f"voice_id not found: {voice_id}")
        try:
            with np.load(voice_path, allow_pickle=False) as profile:
                speaker = torch.from_numpy(profile["speaker"]).float()
                prosody = torch.from_numpy(profile["prosody"]).float()
        except Exception as exc:
            raise VTTSBackendError(f"Voice profile bị hỏng: {voice_id}") from exc
        return speaker, prosody

    @staticmethod
    def _infer_from_embeddings(
        engine: Any,
        text: str,
        speaker: Any,
        prosody: Any,
        *,
        length_scale: float,
        noise_scale: float,
        noise_scale_w: float,
    ) -> tuple[Any, int]:
        """Run V-TTS inference without decoding the reference audio again."""
        import torch
        from src.nn import commons
        from src.text import cleaned_text_to_sequence
        from src.vietnamese.phonemizer import text_to_phonemes
        from src.vietnamese.text_processor import process_vietnamese_text

        processed = process_vietnamese_text(text)
        phones, tones_raw, _ = text_to_phonemes(
            processed, use_viphoneme=os.name != "nt"
        )
        phone_ids, tone_ids, language_ids = cleaned_text_to_sequence(
            phones, tones_raw, "VI"
        )
        if getattr(engine.hps.data, "add_blank", True):
            phone_ids = commons.intersperse(phone_ids, 0)
            tone_ids = commons.intersperse(tone_ids, 0)
            language_ids = commons.intersperse(language_ids, 0)

        device = engine.device
        phone = torch.LongTensor(phone_ids).unsqueeze(0).to(device)
        tone = torch.LongTensor(tone_ids).unsqueeze(0).to(device)
        language = torch.LongTensor(language_ids).unsqueeze(0).to(device)
        phone_length = torch.LongTensor([phone.shape[1]]).to(device)
        bert = torch.zeros(1, 1024, phone.shape[1], device=device)
        ja_bert = torch.zeros(1, 768, phone.shape[1], device=device)
        speaker = speaker.to(device).unsqueeze(-1)
        prosody = prosody.to(device)

        with torch.no_grad():
            output, *_ = engine.model.infer(
                phone,
                phone_length,
                sid=None,
                tone=tone,
                language=language,
                bert=bert,
                ja_bert=ja_bert,
                g=speaker,
                prosody=prosody,
                prosody_predictor=engine.prosody_predictor,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )
        return output[0, 0].cpu().numpy(), engine.sampling_rate

    @staticmethod
    def _encode_wav(audio: Any, sample_rate: int) -> bytes:
        import soundfile as sound_file

        output = BytesIO()
        sound_file.write(
            output,
            audio,
            sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        return output.getvalue()

    def synthesize_registered_voice(
        self,
        text: str,
        voice_id: str,
        *,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
    ) -> bytes:
        """Synthesize speech from a persisted voice profile and text only."""
        engine = self._load_engine()
        speaker, prosody = self._load_voice_embeddings(voice_id)
        try:
            audio, sample_rate = self._infer_from_embeddings(
                engine,
                text,
                speaker,
                prosody,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
            )
            return self._encode_wav(audio, sample_rate)
        except VTTSBackendError:
            raise
        except Exception as exc:
            raise VTTSBackendError(f"V-TTS không tạo được audio: {exc}") from exc

    def synthesize_wav(
        self,
        text: str,
        reference_audio: Path,
        *,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
    ) -> bytes:
        """Synthesize one complete utterance and return a PCM WAV file."""
        engine = self._load_engine()
        try:
            audio, sample_rate = engine.synthesize(
                text=text,
                reference_audio=str(reference_audio),
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
            )
            return self._encode_wav(audio, sample_rate)
        except Exception as exc:
            raise VTTSBackendError(f"V-TTS không tạo được audio: {exc}") from exc


vtts_backend = VTTSBackend()
