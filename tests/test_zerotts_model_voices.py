"""Built-in Ha My must route correctly even before the model is loaded."""

import unicodedata

from fastapi.testclient import TestClient
import pytest

import server
from zerotts_backend import ZeroTTSBackend, ZeroTTSInvalidVoiceIdError

REVISION = "c2bfbd67dc648cac455077333f7cf5c18a2e3bb4"


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("ZEROTTS_REVISION", REVISION)
    for name in ("ZEROTTS_MODEL_DIR", "ZEROTTS_MODEL_ID", "ZEROTTS_BUNDLED_VOICES_DIR"):
        monkeypatch.delenv(name, raising=False)
    backend = ZeroTTSBackend()
    monkeypatch.setattr(backend, "_load_engine", lambda: pytest.fail("Listing/routing must not load weights"))
    return backend


@pytest.mark.parametrize("name", ["hamy", "HAMY", "Hà My", "HàMy", "hà_my", "hà-my", " ha_my ", "ha-my", "Ha My",
                                 unicodedata.normalize("NFD", "Hà My")])
def test_ha_my_alias_is_canonical_and_available_before_inference(backend, name):
    assert backend._validate_voice_id(name) == "hamy"
    assert backend.has_local_voice(name)


@pytest.mark.parametrize("name", ["../hamy", "Hà/My", "bad voice", "hamy;cmd", ""])
def test_unrecognized_unsafe_voice_names_are_still_rejected(backend, name):
    with pytest.raises(ZeroTTSInvalidVoiceIdError):
        backend._validate_voice_id(name)


def test_model_voices_are_listed_without_loading_weights(backend, monkeypatch):
    voices = {item["voice_id"]: item for item in backend.list_voices()}
    assert {"hamy", "maichi", "baotrang", "giahuy", "huuduc", "kimoanh", "quangminh", "tiendat"} <= voices.keys()
    assert voices["hamy"]["source"] == "model"
    assert not backend.is_loaded
    monkeypatch.setenv("ZEROTTS_REVISION", "a" * 40)
    assert not backend._model_voice_catalog()
    monkeypatch.setenv("ZEROTTS_REVISION", REVISION)
    monkeypatch.setenv("ZEROTTS_MODEL_DIR", "custom-model")
    assert not backend._model_voice_catalog()


@pytest.mark.parametrize("method", ["get", "post", "head"])
@pytest.mark.parametrize("field", ["voice", "voice_id", "voiceid"])
def test_ha_my_auto_routes_to_zerotts_on_a_cold_start(backend, monkeypatch, method, field):
    monkeypatch.setattr(server, "FORCE_VOICE_ID", False)
    monkeypatch.setattr(server, "zerotts_backend", backend)
    captured = []

    def synthesize(text, voice, **kwargs):
        captured.append(backend._validate_voice_id(voice))
        return b"RIFF-ha-my"

    monkeypatch.setattr(backend, "synthesize_wav", synthesize)
    monkeypatch.setattr(server.vtts_backend, "synthesize_registered_voice", lambda *a, **kw: pytest.fail("Wrong engine"))
    monkeypatch.setattr(server.vieneu_backend, "synthesize_wav", lambda *a, **kw: pytest.fail("Wrong engine"))
    payload = {"text": "Xin chao", field: "Hà My"}
    client = TestClient(server.app)
    response = (client.post("/tts", json=payload) if method == "post"
                else getattr(client, method)("/tts", params=payload))
    assert response.status_code == 200
    assert captured == ["hamy"]


def test_voice_endpoint_includes_ha_my_before_model_start(backend, monkeypatch):
    monkeypatch.setattr(server, "zerotts_backend", backend)
    response = TestClient(server.app).get("/zerotts/voices")
    assert response.status_code == 200
    assert any(voice["voice_id"] == "hamy" for voice in response.json()["voices"])
