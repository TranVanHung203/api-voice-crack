"""Verify voice override behavior without model loads or network calls."""

import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture(autouse=True)
def routing_configuration(monkeypatch):
    # Exercise each scenario independently of the owner's editable defaults.
    monkeypatch.setattr(server, "FORCE_VOICE_ID", False)
    monkeypatch.setattr(server, "FIXED_PROVIDER", None)


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


@pytest.mark.parametrize("method", ["get", "post", "head"])
@pytest.mark.parametrize("provider", ["edge", "vtts", "zerotts", "vieneu", "gwen"])
def test_fixed_provider_and_voice_override_client_routing(monkeypatch, method, provider):
    captured = []
    monkeypatch.setattr(server, "FORCE_VOICE_ID", True)
    monkeypatch.setattr(server, "FIXED_VOICE_ID", " owner ")
    monkeypatch.setattr(server, "FIXED_PROVIDER", f" {provider.upper()} ")

    def synthesize(engine):
        def run(text, voice_id, **options):
            captured.append((engine, voice_id))
            return b"audio"
        return run

    async def edge(text, voice, rate, volume, pitch):
        captured.append(("edge", voice))
        return b"audio"

    monkeypatch.setattr(server, "_get_audio", edge)
    monkeypatch.setattr(server.vtts_backend, "synthesize_registered_voice", synthesize("vtts"))
    for name in ("zerotts", "vieneu", "gwen"):
        monkeypatch.setattr(getattr(server, name + "_backend"), "synthesize_wav", synthesize(name))
    payload = {"text": "Hello", "provider": "invalid-client-provider", "voice": "client",
               "voiceid": "a", "voice_id": "b"}
    response = getattr(TestClient(server.app), method)(
        "/tts", **{"json" if method == "post" else "params": payload})
    assert response.status_code == 200
    assert captured == [(provider, "owner")]


@pytest.mark.parametrize("provider", ["unknown", "", "   "])
def test_invalid_fixed_provider_reports_configuration_error(monkeypatch, provider):
    monkeypatch.setattr(server, "FORCE_VOICE_ID", True)
    monkeypatch.setattr(server, "FIXED_PROVIDER", provider)
    response = TestClient(server.app).get("/tts", params={"text": "Hello", "provider": "edge"})
    assert response.status_code == 503
    assert "FIXED_PROVIDER" in response.json()["detail"]


def test_fixed_provider_is_ignored_when_override_disabled(monkeypatch):
    captured = []
    monkeypatch.setattr(server, "FORCE_VOICE_ID", False)
    monkeypatch.setattr(server, "FIXED_PROVIDER", "gwen")

    async def edge(text, voice, rate, volume, pitch):
        captured.append(voice)
        return b"audio"

    monkeypatch.setattr(server, "_get_audio", edge)
    response = TestClient(server.app).get(
        "/tts", params={"text": "Hello", "provider": "edge", "voice": "client"})
    assert response.status_code == 200
    assert captured == ["client"]
