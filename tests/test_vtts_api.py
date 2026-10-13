"""API tests for the non-streaming V-TTS endpoint."""

from pathlib import Path

from fastapi.testclient import TestClient

import server
from vtts_backend import VTTSBackendError, VTTSVoiceNotFoundError


client = TestClient(server.app)


def test_vtts_clone_returns_wav_and_removes_upload(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_synthesize(text: str, reference_audio: Path, **options: float) -> bytes:
        captured["text"] = text
        captured["reference_bytes"] = reference_audio.read_bytes()
        captured["options"] = options
        return b"RIFF-fake-wav"

    monkeypatch.setattr(server, "_VTTS_TEMP_DIR", tmp_path)
    monkeypatch.setattr(server.vtts_backend, "synthesize_wav", fake_synthesize)

    response = client.post(
        "/vtts/clone",
        data={
            "text": "  Xin chào  ",
            "length_scale": "0.9",
            "noise_scale": "0.5",
            "noise_scale_w": "0.7",
        },
        files={"reference_audio": ("voice.wav", b"reference", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFF-fake-wav"
    assert captured == {
        "text": "Xin chào",
        "reference_bytes": b"reference",
        "options": {
            "length_scale": 0.9,
            "noise_scale": 0.5,
            "noise_scale_w": 0.7,
        },
    }
    assert list(tmp_path.iterdir()) == []


def test_vtts_clone_rejects_unsupported_reference_file() -> None:
    response = client.post(
        "/vtts/clone",
        data={"text": "Xin chào"},
        files={"reference_audio": ("voice.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 400
    assert "reference_audio" in response.json()["detail"]


def test_vtts_clone_reports_backend_error(monkeypatch, tmp_path: Path) -> None:
    def fail_synthesis(*_args, **_kwargs) -> bytes:
        raise VTTSBackendError("model unavailable")

    monkeypatch.setattr(server, "_VTTS_TEMP_DIR", tmp_path)
    monkeypatch.setattr(server.vtts_backend, "synthesize_wav", fail_synthesis)

    response = client.post(
        "/vtts/clone",
        data={"text": "Xin chào"},
        files={"reference_audio": ("voice.wav", b"reference", "audio/wav")},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "model unavailable"}
    assert list(tmp_path.iterdir()) == []


def test_register_voice_returns_reusable_id(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_register(reference_audio: Path, voice_id: str | None) -> str:
        captured["reference_bytes"] = reference_audio.read_bytes()
        captured["voice_id"] = voice_id
        return voice_id or "generated-id"

    monkeypatch.setattr(server, "_VTTS_TEMP_DIR", tmp_path)
    monkeypatch.setattr(server.vtts_backend, "register_voice", fake_register)

    response = client.post(
        "/vtts/voices",
        data={"voice_id": "hung"},
        files={"reference_audio": ("voice.wav", b"reference", "audio/wav")},
    )

    assert response.status_code == 201
    assert response.json() == {"voice_id": "hung"}
    assert captured == {"reference_bytes": b"reference", "voice_id": "hung"}
    assert list(tmp_path.iterdir()) == []


def test_synthesize_registered_voice_needs_only_text_and_id(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_synthesize(text: str, voice_id: str, **options: float) -> bytes:
        captured.update(text=text, voice_id=voice_id, options=options)
        return b"RIFF-registered-voice"

    monkeypatch.setattr(
        server.vtts_backend, "synthesize_registered_voice", fake_synthesize
    )

    response = client.post(
        "/vtts/synthesize",
        json={"text": "  Xin chào  ", "voice_id": "hung"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFF-registered-voice"
    assert captured == {
        "text": "Xin chào",
        "voice_id": "hung",
        "options": {
            "length_scale": 1.0,
            "noise_scale": 0.667,
            "noise_scale_w": 0.8,
        },
    }


def test_synthesize_registered_voice_returns_404(monkeypatch) -> None:
    def fail_synthesis(*_args, **_kwargs) -> bytes:
        raise VTTSVoiceNotFoundError("voice_id not found: missing")

    monkeypatch.setattr(
        server.vtts_backend, "synthesize_registered_voice", fail_synthesis
    )

    response = client.post(
        "/vtts/synthesize",
        json={"text": "Xin chào", "voice_id": "missing"},
    )

    assert response.status_code == 404


def test_get_tts_uses_registered_voice_when_voiceid_is_present(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_synthesize(text: str, voice_id: str, **_options: float) -> bytes:
        captured.update(text=text, voice_id=voice_id)
        return b"RIFF-get-route"

    monkeypatch.setattr(
        server.vtts_backend, "synthesize_registered_voice", fake_synthesize
    )

    response = client.get(
        "/tts",
        params={"text": "  Xin chào  ", "voiceid": "hung"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFF-get-route"
    assert captured == {"text": "Xin chào", "voice_id": "hung"}


def test_get_tts_keeps_edge_tts_behavior_without_voiceid(monkeypatch) -> None:
    async def fake_get_audio(*_args, **_kwargs) -> bytes:
        return b"edge-mp3"

    monkeypatch.setattr(server, "_get_audio", fake_get_audio)

    response = client.get("/tts", params={"text": "Xin chào"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"edge-mp3"


def test_post_tts_uses_registered_vtts_voice(monkeypatch) -> None:
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        server.zerotts_backend, "has_local_voice", lambda _voice_id: False
    )

    def fake_synthesize(text: str, voice_id: str, **_options: float) -> bytes:
        captured.update(text=text, voice_id=voice_id)
        return b"RIFF-post-vtts"

    monkeypatch.setattr(
        server.vtts_backend, "synthesize_registered_voice", fake_synthesize
    )

    response = client.post(
        "/tts",
        json={"text": "  Xin chao  ", "voiceid": "hung", "provider": "vtts"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFF-post-vtts"
    assert captured == {"text": "Xin chao", "voice_id": "hung"}


def test_post_tts_keeps_edge_tts_behavior_without_voiceid(monkeypatch) -> None:
    captured: dict[str, str] = {}

    async def fake_get_audio(
        text: str, voice: str, rate: str, volume: str, pitch: str
    ) -> bytes:
        captured.update(
            text=text, voice=voice, rate=rate, volume=volume, pitch=pitch
        )
        return b"edge-post-mp3"

    monkeypatch.setattr(server, "_get_audio", fake_get_audio)

    response = client.post(
        "/tts",
        json={
            "text": "Xin chao",
            "voice": "vi-VN-HoaiMyNeural",
            "rate": "-5%",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"edge-post-mp3"
    assert captured == {
        "text": "Xin chao",
        "voice": "vi-VN-HoaiMyNeural",
        "rate": "-5%",
        "volume": "+100%",
        "pitch": "+0Hz",
    }
