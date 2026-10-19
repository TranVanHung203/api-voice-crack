"""VieNeu routing, HTTP audio delivery, and serialized model access."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import sys
import time
from types import SimpleNamespace
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from vieneu_backend import VieNeuBackend, VieNeuBackendError, VieNeuVoiceNotFoundError


client = TestClient(server.app)


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("VIENEU_AUDIO_CACHE_DIR", str(tmp_path))


@pytest.mark.parametrize("method", ["get", "post"])
@pytest.mark.parametrize("voice_field", ["voice", "voiceid", "voice_id", None])
def test_vieneu_routes_without_touching_other_backends(monkeypatch, method, voice_field):
    captured = []

    def synthesize(text, voice_id):
        captured.append((text, voice_id))
        return b"RIFF-vieneu"

    def fail(*args, **kwargs):
        pytest.fail("Explicit VieNeu requests must not access another backend")

    monkeypatch.setattr(server.vieneu_backend, "synthesize_wav", synthesize)
    monkeypatch.setattr(server.zerotts_backend, "has_local_voice", fail)
    monkeypatch.setattr(server.vtts_backend, "synthesize_registered_voice", fail)
    payload = {"text": "  Xin chào  ", "provider": " VieNeu "}
    if voice_field:
        payload[voice_field] = "Mai Anh"
    response = getattr(client, method)("/tts", **{"params" if method == "get" else "json": payload})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFF-vieneu"
    assert captured == [("Xin chào", "Mai Anh" if voice_field else None)]


def test_vieneu_mp3_and_range_delivery(monkeypatch, tmp_path):
    monkeypatch.setattr(server.vieneu_backend, "synthesize_wav", lambda *args: b"wav")
    monkeypatch.setattr(server.vieneu_backend, "audio_cache_dir", lambda: tmp_path)

    def encode(audio, directory):
        assert audio == b"wav" and directory == tmp_path
        return b"ID3-test-mp3"

    monkeypatch.setattr(server, "encode_mp3", encode)
    params = {"text": "Hello", "provider": "vieneu", "format": "mp3"}
    response = client.get("/tts", params=params, headers={"Range": "bytes=0-2"})
    assert response.status_code == 206
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"ID3"
    head = client.head("/tts", params=params)
    assert head.content == b""
    assert head.headers["content-length"] == str(len(b"ID3-test-mp3"))


@pytest.mark.parametrize("error,status", [
    (VieNeuVoiceNotFoundError("unknown voice"), 404),
    (VieNeuBackendError("model unavailable"), 503),
])
def test_vieneu_errors(monkeypatch, error, status):
    def fail(*args):
        raise error

    monkeypatch.setattr(server.vieneu_backend, "synthesize_wav", fail)
    response = client.post("/tts", json={"text": "Hello", "provider": "vieneu"})
    assert response.status_code == status
    assert response.json()["detail"] == str(error)


def test_vieneu_fixed_voice_and_conflicting_aliases(monkeypatch):
    captured = []

    def synthesize(text, voice_id):
        captured.append(voice_id)
        return b"RIFF"

    monkeypatch.setattr(server.vieneu_backend, "synthesize_wav", synthesize)
    payload = {"text": "Hello", "provider": "vieneu", "voiceid": "a", "voice_id": "b"}
    assert client.get("/tts", params=payload).status_code == 400
    assert client.post("/tts", json=payload).status_code == 400
    monkeypatch.setattr(server, "FORCE_VOICE_ID", True)
    monkeypatch.setattr(server, "FIXED_VOICE_ID", "Mai Anh")
    assert client.post("/tts", json=payload).status_code == 200
    assert captured == ["Mai Anh"]


def test_vieneu_voice_list(monkeypatch):
    voices = [{"voice_id": "Mai Anh", "label": "Mai Anh", "provider": "vieneu"}]
    monkeypatch.setattr(server.vieneu_backend, "list_voices", lambda: voices)
    assert client.get("/vieneu/voices").json() == {"voices": voices}


def test_vieneu_is_lazy_and_loads_once_across_threads(monkeypatch):
    loads, active, maximum = [], [], []

    def infer(text, **options):
        active.append(text)
        maximum.append(len(active))
        time.sleep(0.01)
        active.remove(text)
        return np.zeros(480, dtype=np.float32)

    def factory(**options):
        loads.append(options)
        return SimpleNamespace(
            infer=infer, sample_rate=48000,
            resolve_voice_name=lambda name: name if name == "Mai Anh" else None,
        )

    monkeypatch.setitem(sys.modules, "vieneu", SimpleNamespace(Vieneu=factory))
    backend = VieNeuBackend()
    assert not loads
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(lambda text: backend.synthesize_wav(text, "Mai Anh"), ["a", "b", "c"]))
    assert len(loads) == 1 and max(maximum) == 1
    with wave.open(BytesIO(outputs[0])) as wav:
        assert (wav.getframerate(), wav.getnframes(), wav.getnchannels()) == (48000, 480, 1)
    with pytest.raises(VieNeuVoiceNotFoundError):
        backend.synthesize_wav("Hello", "unknown")


@pytest.mark.parametrize("audio", [np.array([]), np.array([float("nan")])])
def test_vieneu_rejects_invalid_model_output(monkeypatch, audio):
    backend = VieNeuBackend()
    backend._engine = SimpleNamespace(infer=lambda *args, **kwargs: audio)
    with pytest.raises(VieNeuBackendError, match="invalid audio"):
        backend.synthesize_wav("Hello")


def test_vieneu_dependency_error_does_not_break_health(monkeypatch):
    monkeypatch.setitem(sys.modules, "vieneu", None)
    with pytest.raises(VieNeuBackendError, match="runtime missing"):
        VieNeuBackend().synthesize_wav("Hello")
    assert client.get("/health").status_code == 200


def test_vieneu_cached_audio_is_stable_for_ranges_and_survives_restart(monkeypatch, tmp_path):
    calls = []

    def infer(*args, **kwargs):
        calls.append(1)
        return np.full(480, len(calls) / 10, dtype=np.float32)

    backend = VieNeuBackend()
    backend._engine = SimpleNamespace(infer=infer, sample_rate=48000)
    monkeypatch.setattr(server, "vieneu_backend", backend)
    params = {"text": "Same sentence", "provider": "vieneu"}
    first = client.get("/tts", params=params)
    part = client.get("/tts", params=params, headers={"Range": "bytes=44-99"})
    assert part.status_code == 206 and part.content == first.content[44:100]
    assert client.get("/tts", params=params, headers={"If-None-Match": first.headers["etag"]}).status_code == 304
    # A fresh backend must read persisted audio without loading the model.
    restarted = VieNeuBackend()
    monkeypatch.setattr(restarted, "_load_engine", lambda: pytest.fail("Cache hit must not load model"))
    assert restarted.synthesize_wav("Same sentence") == first.content
    assert len(calls) == 1 and len(list(tmp_path.glob("*.wav"))) == 1


def test_vieneu_ram_cache_is_bounded():
    backend = VieNeuBackend()
    backend._ram_limit = 10
    backend._remember("a", b"a" * 6)
    backend._remember("b", b"b" * 6)
    backend._remember("large", b"x" * 11)
    assert list(backend._audio_cache) == ["b"]
