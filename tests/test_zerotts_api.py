"""API tests for the ZeroTTS integration."""

import wave
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
import zerotts_backend
from zerotts_backend import ZeroTTSBackend, ZeroTTSVoiceNotFoundError


client = TestClient(server.app)


@pytest.fixture(autouse=True)
def isolated_audio_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(ZeroTTSBackend, "_audio_cache_dir", classmethod(lambda cls: tmp_path))


def test_zerotts_wav_encoder_accepts_model_batch_dimension() -> None:
    audio = np.zeros((1, 4800), dtype=np.float32)

    encoded = ZeroTTSBackend._encode_wav(audio, 48_000)

    with wave.open(BytesIO(encoded), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 48_000
        assert wav_file.getnframes() == 4800


def test_zerotts_synthesize_returns_wav(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_synthesize(text: str, voice_id: str, **options: object) -> bytes:
        captured.update(text=text, voice_id=voice_id, options=options)
        return b"RIFF-zerotts"

    monkeypatch.setattr(server.zerotts_backend, "synthesize_wav", fake_synthesize)

    response = client.post(
        "/zerotts/synthesize",
        json={
            "text": "  Xin chao  ",
            "voice_id": "VpYLlCQEgcoHgycJstNP",
            "cfg_scale": 1.2,
            "audio_temperature": 0.7,
            "audio_repetition_penalty": 1.3,
            "max_chunk_sec": 10,
            "max_frames": 500,
            "eoa_extra_frames": 4,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFF-zerotts"
    assert captured == {
        "text": "Xin chao",
        "voice_id": "VpYLlCQEgcoHgycJstNP",
        "options": {
            "cfg_scale": 1.2,
            "audio_temperature": 0.7,
            "audio_repetition_penalty": 1.3,
            "max_chunk_sec": 10.0,
            "max_frames": 500,
            "eoa_extra_frames": 4,
        },
    }


def test_zerotts_synthesize_returns_404(monkeypatch) -> None:
    def fail_synthesis(*_args, **_kwargs) -> bytes:
        raise ZeroTTSVoiceNotFoundError("ZeroTTS voice_id not found: missing")

    monkeypatch.setattr(server.zerotts_backend, "synthesize_wav", fail_synthesis)

    response = client.post(
        "/zerotts/synthesize",
        json={"text": "Xin chao", "voice_id": "missing"},
    )

    assert response.status_code == 404


def test_get_tts_auto_detects_installed_zerotts_voice(monkeypatch) -> None:
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        server.zerotts_backend,
        "has_local_voice",
        lambda voice_id: voice_id == "VpYLlCQEgcoHgycJstNP",
    )

    def fake_synthesize(text: str, voice_id: str, **_options: object) -> bytes:
        captured.update(text=text, voice_id=voice_id)
        return b"RIFF-auto-zerotts"

    monkeypatch.setattr(server.zerotts_backend, "synthesize_wav", fake_synthesize)

    response = client.get(
        "/tts",
        params={"text": "  Xin chao  ", "voiceid": "VpYLlCQEgcoHgycJstNP"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFF-auto-zerotts"
    assert captured == {
        "text": "Xin chao",
        "voice_id": "VpYLlCQEgcoHgycJstNP",
    }


def test_post_tts_uses_zerotts_with_voiceid(monkeypatch) -> None:
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        server.zerotts_backend,
        "has_local_voice",
        lambda voice_id: voice_id == "VpYLlCQEgcoHgycJstNP",
    )

    def fake_synthesize(text: str, voice_id: str, **_options: object) -> bytes:
        captured.update(text=text, voice_id=voice_id)
        return b"RIFF-post-zerotts"

    monkeypatch.setattr(server.zerotts_backend, "synthesize_wav", fake_synthesize)

    response = client.post(
        "/tts",
        json={
            "text": "  Xin chao  ",
            "voiceid": "VpYLlCQEgcoHgycJstNP",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFF-post-zerotts"
    assert captured == {
        "text": "Xin chao",
        "voice_id": "VpYLlCQEgcoHgycJstNP",
    }


def test_post_tts_auto_detects_zerotts_id_in_voice_field(monkeypatch) -> None:
    monkeypatch.setattr(
        server.zerotts_backend,
        "has_local_voice",
        lambda voice_id: voice_id == "VpYLlCQEgcoHgycJstNP",
    )
    monkeypatch.setattr(
        server.zerotts_backend,
        "synthesize_wav",
        lambda *_args, **_kwargs: b"RIFF-post-voice-field",
    )

    response = client.post(
        "/tts",
        json={
            "text": "Xin chao",
            "voice": "VpYLlCQEgcoHgycJstNP",
        },
    )

    assert response.status_code == 200
    assert response.content == b"RIFF-post-voice-field"


def test_post_tts_accepts_voice_id_alias(monkeypatch) -> None:
    monkeypatch.setattr(
        server.zerotts_backend,
        "has_local_voice",
        lambda voice_id: voice_id == "VpYLlCQEgcoHgycJstNP",
    )
    monkeypatch.setattr(
        server.zerotts_backend,
        "synthesize_wav",
        lambda *_args, **_kwargs: b"RIFF-post-alias",
    )

    response = client.post(
        "/tts",
        json={
            "text": "Xin chao",
            "voice_id": "VpYLlCQEgcoHgycJstNP",
            "provider": "zerotts",
        },
    )

    assert response.status_code == 200
    assert response.content == b"RIFF-post-alias"


def test_get_tts_provider_can_force_zerotts(monkeypatch) -> None:
    monkeypatch.setattr(
        server.zerotts_backend,
        "has_local_voice",
        lambda _voice_id: False,
    )
    monkeypatch.setattr(
        server.zerotts_backend,
        "synthesize_wav",
        lambda *_args, **_kwargs: b"RIFF-forced-zerotts",
    )

    response = client.get(
        "/tts",
        params={"text": "Xin chao", "voiceid": "maichi", "provider": "zerotts"},
    )

    assert response.status_code == 200
    assert response.content == b"RIFF-forced-zerotts"


def test_list_zerotts_voices(monkeypatch) -> None:
    voices = [
        {
            "voice_id": "VpYLlCQEgcoHgycJstNP",
            "display_name": "chi tham",
            "language": "vi",
            "description": "Female, Middle-Age, Southern",
            "tags": ["female", "middle-age", "southern"],
            "source": "installed",
        }
    ]
    monkeypatch.setattr(server.zerotts_backend, "list_voices", lambda: voices)

    response = client.get("/zerotts/voices")

    assert response.status_code == 200
    assert response.json() == {"voices": voices}


def test_get_tts_rejects_unknown_provider() -> None:
    response = client.get(
        "/tts", params={"text": "Xin chao", "provider": "unknown"}
    )

    assert response.status_code == 400


def test_zerotts_web_defaults_and_repeatability(monkeypatch) -> None:
    monkeypatch.setattr(zerotts_backend, "_SAMPLING_SEED", 28)
    calls = []

    def generate(text, **options):
        calls.append((text, options))
        return np.random.random((1, 4800)).astype(np.float32)

    backend = ZeroTTSBackend()
    monkeypatch.setattr(backend, "_load_engine", lambda: SimpleNamespace(
        sample_rate=48_000, synthesize=generate
    ))
    before = np.random.get_state()
    first = backend.synthesize_wav("Kh\u00e1m ph\u00e1", "hamy")
    assert first == backend.synthesize_wav("Kh\u00e1m ph\u00e1", "hamy")
    assert calls[0] == ("Kh\u00e1m ph\u00e1.", {
        "voice": "hamy", "cfg_scale": 1.0, "audio_temperature": 0.8,
        "audio_repetition_penalty": 1.2, "max_frames": 1500, "eoa_extra_frames": 1,
    })
    expected = np.random.RandomState(28).random((1, 4800)).astype(np.float32)
    assert first == backend._encode_wav(expected, 48_000)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_zerotts_splits_long_text_without_losing_words(monkeypatch) -> None:
    calls = []

    def generate(text, **_options):
        calls.append(text)
        return np.zeros((1, 4800), dtype=np.float32)

    backend = ZeroTTSBackend()
    monkeypatch.setattr(backend, "_load_engine", lambda: SimpleNamespace(
        sample_rate=48_000, synthesize=generate
    ))
    text = " ".join(["Kh\u00e1m ph\u00e1"] * 40)
    audio = backend.synthesize_wav(text, "hamy")
    assert len(calls) > 1
    assert " ".join(c.rstrip(".") for c in calls) == text
    with wave.open(BytesIO(audio)) as w:
        assert w.getnframes() == len(calls) * 4800 + (len(calls) - 1) * 15360
