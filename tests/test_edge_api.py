"""Edge routing and actionable error responses."""

import aiohttp
from collections import OrderedDict
import pytest
from fastapi.testclient import TestClient
from edge_tts.exceptions import NoAudioReceived

import server


client = TestClient(server.app)


def test_edge_provider_routes_to_edge_with_voice_id_present(monkeypatch):
    async def fake_audio(text, voice, rate, volume, pitch):
        assert voice == "vi-VN-HoaiMyNeural"
        return b"edge-audio"

    def unexpected_zerotts(*args, **kwargs):
        pytest.fail("Explicit Edge provider must not call ZeroTTS")

    monkeypatch.setattr(server, "_get_audio", fake_audio)
    monkeypatch.setattr(server.zerotts_backend, "synthesize_wav", unexpected_zerotts)
    response = client.post("/tts", json={
        "text": "Xin chao", "provider": "edge", "voice_id": "hamy"
    })
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"


@pytest.mark.parametrize("error,status", [
    (ValueError("Invalid pitch ' 0Hz'."), 400),
    (NoAudioReceived("No audio was received."), 502),
    (aiohttp.ClientConnectionError("Connection failed"), 502),
    (TimeoutError("Timed out"), 502),
])
def test_edge_errors_include_cause(monkeypatch, error, status):
    async def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(server, "_get_audio", fail)
    response = client.get("/tts", params={"text": "Xin chao", "provider": "edge"})
    assert response.status_code == status
    assert str(error) in response.json()["detail"]


@pytest.mark.parametrize("recover", [True, False])
def test_edge_retries_missing_audio_and_caches_only_success(monkeypatch, tmp_path, recover):
    attempts = []

    class FakeCommunicate:
        def __init__(self, **options):
            attempts.append(options)

        async def stream(self):
            if not recover or len(attempts) < 3:
                raise NoAudioReceived("No audio was received")
            yield {"type": "audio", "data": b"edge-success"}

    async def no_sleep(*args):
        pass

    monkeypatch.setattr(server.edge_tts, "Communicate", FakeCommunicate)
    monkeypatch.setattr(server.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(server, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(server, "_audio_cache", OrderedDict())
    monkeypatch.setattr(server, "_cache_locks", {})
    response = client.get("/tts", params={"text": "Xin chao", "provider": "edge"})
    assert len(attempts) == 3
    assert all(a == attempts[0] for a in attempts)
    assert response.status_code == (200 if recover else 502)
    assert len(list(tmp_path.glob("*.mp3"))) == (1 if recover else 0)
    if recover:
        assert response.content == b"edge-success"
        assert client.get("/tts", params={"text": "Xin chao", "provider": "edge"}).status_code == 200
        assert len(attempts) == 3
