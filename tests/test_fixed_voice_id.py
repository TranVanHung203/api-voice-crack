"""Verify voice override behavior without model loads or network calls."""

import pytest
from fastapi.testclient import TestClient

import server


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("endpoint", ["get", "post", "vtts", "zerotts"])
def test_voice_override(monkeypatch, enabled, endpoint):
    captured = []

    def synthesize(text, voice_id, **options):
        captured.append(voice_id)
        return b"RIFF-test"

    monkeypatch.setattr(server, "FORCE_VOICE_ID", enabled)
    monkeypatch.setattr(server, "FIXED_VOICE_ID", " owner ")
    monkeypatch.setattr(server.zerotts_backend, "has_local_voice", lambda _: True)
    monkeypatch.setattr(server.zerotts_backend, "synthesize_wav", synthesize)
    monkeypatch.setattr(server.vtts_backend, "synthesize_registered_voice", synthesize)
    client = TestClient(server.app)
    if endpoint == "get":
        response = client.get("/tts", params={"text": "Hello", "voiceid": "client"})
    elif endpoint == "post":
        response = client.post("/tts", json={"text": "Hello", "voice_id": "client"})
    else:
        response = client.post(
            f"/{endpoint}/synthesize", json={"text": "Hello", "voice_id": "client"}
        )
    assert response.status_code == 200
    assert captured == ["owner" if enabled else "client"]


@pytest.mark.parametrize("payload", [
    {"voiceid": "a", "voice_id": "b", "provider": "edge"},
    {},
])
def test_fixed_voice_cannot_be_bypassed(monkeypatch, payload):
    captured = []
    monkeypatch.setattr(server, "FORCE_VOICE_ID", True)
    monkeypatch.setattr(server, "FIXED_VOICE_ID", "owner")
    monkeypatch.setattr(server.zerotts_backend, "has_local_voice", lambda v: v == "owner")

    def synthesize(text, voice_id, **options):
        captured.append(voice_id)
        return b"RIFF-test"

    monkeypatch.setattr(server.zerotts_backend, "synthesize_wav", synthesize)
    response = TestClient(server.app).post("/tts", json={"text": "Hello", **payload})
    assert response.status_code == 200
    assert captured == ["owner"]


def test_empty_fixed_voice_reports_configuration_error(monkeypatch):
    monkeypatch.setattr(server, "FORCE_VOICE_ID", True)
    monkeypatch.setattr(server, "FIXED_VOICE_ID", "  ")
    response = TestClient(server.app).get("/tts", params={"text": "Hello"})
    assert response.status_code == 503
    assert "FIXED_VOICE_ID" in response.json()["detail"]


def test_fixed_model_voice_preserves_zerotts_provider(monkeypatch):
    captured = []
    monkeypatch.setattr(server, "FORCE_VOICE_ID", True)
    monkeypatch.setattr(server, "FIXED_VOICE_ID", "hamy")
    monkeypatch.setattr(server.zerotts_backend, "has_local_voice", lambda _: False)

    def synthesize(text, voice_id, **options):
        captured.append(voice_id)
        return b"RIFF-hamy"

    monkeypatch.setattr(server.zerotts_backend, "synthesize_wav", synthesize)
    response = TestClient(server.app).get(
        "/tts", params={"text": "Hello", "voiceid": "maichi", "provider": "zerotts"}
    )
    assert response.status_code == 200
    assert captured == ["hamy"]
