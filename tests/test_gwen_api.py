"""Gwen routing, stable audio delivery, worker isolation, and text chunking."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO, StringIO
import json
from pathlib import Path
import queue
import sys
import time
from types import SimpleNamespace
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

import gwen_backend as module
from gwen_backend import GwenBackend, GwenBackendError, GwenVoiceNotFoundError, resolve_voice
import gwen_worker
import server


def wav_bytes(value=0):
    out = BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(np.full(2400, value, dtype=np.int16).tobytes())
    return out.getvalue()


@pytest.fixture
def backend(monkeypatch, tmp_path):
    monkeypatch.setenv("GWEN_AUDIO_CACHE_DIR", str(tmp_path))
    backend = GwenBackend()
    monkeypatch.setattr(server, "gwen_backend", backend)
    yield backend
    backend.close()


@pytest.mark.parametrize("method", ["get", "post"])
@pytest.mark.parametrize("field", ["voice", "voiceid", "voice_id", None])
def test_gwen_unified_route(backend, monkeypatch, method, field):
    captured = []

    def synthesize(text, voice_id):
        captured.append((text, voice_id))
        return wav_bytes()

    def fail(*args, **kwargs):
        pytest.fail("Gwen must not route into another engine")

    monkeypatch.setattr(backend, "synthesize_wav", synthesize)
    monkeypatch.setattr(server.zerotts_backend, "has_local_voice", fail)
    monkeypatch.setattr(server.vieneu_backend, "synthesize_wav", fail)
    payload = {"text": "  Xin chào  ", "provider": " Gwen "}
    if field:
        payload[field] = "Yến Nhi"
    response = getattr(TestClient(server.app), method)(
        "/tts", **{"params" if method == "get" else "json": payload})
    assert response.status_code == 200 and response.headers["content-type"] == "audio/wav"
    assert captured == [("Xin chào", "Yến Nhi" if field else None)]


def test_gwen_cached_mp3_get_head_range_and_etag(backend, monkeypatch):
    calls = []

    def worker(text, voice_id, path):
        calls.append((text, voice_id))
        path.write_bytes(wav_bytes(len(calls)))

    monkeypatch.setattr(backend, "_run_worker", worker)
    monkeypatch.setattr(server, "encode_mp3", lambda audio, cache: b"ID3" + audio)
    client = TestClient(server.app)
    params = {"text": "Hello", "provider": "gwen", "voice": "Yến Nhi", "format": "mp3"}
    first = client.get("/tts", params=params)
    assert first.status_code == 200 and first.headers["content-type"] == "audio/mpeg"
    part = client.get("/tts", params=params, headers={"Range": "bytes=0-2"})
    assert part.status_code == 206 and part.content == first.content[:3] == b"ID3"
    head = client.head("/tts", params=params)
    assert head.content == b"" and int(head.headers["content-length"]) == len(first.content)
    assert client.get("/tts", params=params, headers={"If-None-Match": first.headers["etag"]}).status_code == 304
    # Aliases share both in-memory and persistent WAV cache entries.
    restarted = GwenBackend()
    monkeypatch.setattr(restarted, "_run_worker", lambda *args: pytest.fail("Disk hit must not start a worker"))
    assert restarted.synthesize_wav("Hello", "yen_nhi") == first.content[3:]
    assert calls == [("Hello", "yen_nhi")]


def test_gwen_voice_list_needs_no_runtime(backend, monkeypatch):
    monkeypatch.setattr(backend, "_start_worker", lambda: pytest.fail("Voice listing must not start model"))
    voices = TestClient(server.app).get("/gwen/voices").json()["voices"]
    assert len(voices) == 9
    assert next(v for v in voices if v["voice_id"] == "yen_nhi")["name"] == "Yến Nhi"
    assert resolve_voice("Yến Nhi")[0] == "yen_nhi"
    assert resolve_voice("YEN-NHI")[0] == "yen_nhi"


def test_gwen_errors_and_alias_conflict(backend, monkeypatch):
    client = TestClient(server.app)
    assert client.get("/tts", params={"provider": "gwen", "text": "Hello", "voice": "unknown"}).status_code == 404
    assert client.get("/tts", params={"provider": "gwen", "text": " "}).status_code == 400
    assert client.get("/tts", params={"provider": "gwen", "text": "Hello", "voiceid": "a", "voice_id": "b"}).status_code == 400
    monkeypatch.setattr(backend, "_run_worker", lambda *args: (_ for _ in ()).throw(GwenBackendError("runtime unavailable")))
    response = client.get("/tts", params={"provider": "gwen", "text": "Hello"})
    assert response.status_code == 503 and "runtime unavailable" in response.json()["detail"]


def test_gwen_fixed_voice_cannot_be_bypassed(backend, monkeypatch):
    calls = []
    monkeypatch.setattr(server, "FORCE_VOICE_ID", True)
    monkeypatch.setattr(server, "FIXED_VOICE_ID", "my_van")

    def worker(text, voice_id, path):
        calls.append(voice_id)
        path.write_bytes(wav_bytes())

    monkeypatch.setattr(backend, "_run_worker", worker)
    response = TestClient(server.app).get("/tts", params={"text": "Hello", "provider": "gwen", "voice": "Yến Nhi"})
    assert response.status_code == 200 and calls == ["my_van"]


def test_gwen_concurrent_misses_are_serialized_and_coalesced(backend, monkeypatch):
    active, maximum, calls = [], [], []

    def worker(text, voice_id, path):
        calls.append(text)
        active.append(text)
        maximum.append(len(active))
        time.sleep(0.01)
        path.write_bytes(wav_bytes())
        active.remove(text)

    monkeypatch.setattr(backend, "_run_worker", worker)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(backend.synthesize_wav, ["Hello", "Hello", "Other"]))
    assert results[0] == results[1] and len(calls) == 2 and max(maximum) == 1


def test_gwen_bad_worker_output_is_not_cached(backend, monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "_run_worker", lambda text, voice_id, path: path.write_bytes(b"RIFFbad"))
    with pytest.raises(GwenBackendError, match="truncated"):
        backend.synthesize_wav("Hello")
    assert not list(tmp_path.iterdir()) and not backend._audio_cache


def test_gwen_worker_timeout_is_stopped(backend, monkeypatch, tmp_path):
    stopped = []
    monkeypatch.setattr(backend, "_start_worker", lambda: None)
    monkeypatch.setattr(backend, "_stop_worker", lambda: stopped.append(True))
    monkeypatch.setenv("GWEN_TIMEOUT", "0.01")
    backend._process = SimpleNamespace(stdin=StringIO())
    backend._responses = queue.Queue()
    with pytest.raises(GwenBackendError, match="timed out"):
        backend._run_worker("Hello", "yen_nhi", tmp_path / "test.wav")
    assert stopped == [True]


def test_gwen_missing_runtime_has_actionable_error(backend, monkeypatch):
    monkeypatch.setenv("GWEN_PYTHON", "Z:/nonexistent/gwen/python.exe")
    with pytest.raises(GwenBackendError, match="setup_gwen.ps1"):
        backend.synthesize_wav("Hello")


def test_gwen_chunking_preserves_words_and_limits_length():
    text = "Xin chào. " + "Một đoạn văn dài cần chia thành nhiều phần. " * 30
    chunks = gwen_worker.text_chunks(text)
    assert " ".join(chunks).split() == text.split()
    assert all(len(c) <= 320 for c in chunks)
    assert all(len(c) <= 320 for c in gwen_worker.text_chunks("x" * 1000))


def test_worker_reuses_reference_prompt_and_encodes_all_chunks(tmp_path):
    prompts, requests = [], []

    def create(**kwargs):
        prompts.append(kwargs)
        return "prompt"

    def generate(**kwargs):
        requests.append(kwargs)
        return [np.ones(240, dtype=np.float32) * 0.1], 24000

    engine = gwen_worker.GwenEngine.__new__(gwen_worker.GwenEngine)
    engine.model = SimpleNamespace(create_voice_clone_prompt=create, generate_voice_clone=generate)
    engine.normalizer = SimpleNamespace(normalize=lambda text: text)
    engine.prompts = {}
    path = tmp_path / "audio.wav"
    engine.synthesize("Xin chào. Tạm biệt!", "yen_nhi", path)
    assert len(prompts) == 1 and len(requests) == 2
    assert all(r["voice_clone_prompt"] == "prompt" and "language" not in r for r in requests)
    with wave.open(str(path)) as wav:
        assert wav.getnframes() == 240 * 2 + round(24000 * 0.15)


def test_health_and_edge_work_when_gwen_runtime_is_missing(backend, monkeypatch):
    monkeypatch.setattr(backend, "_start_worker", lambda: pytest.fail("Unrelated requests must not load Gwen"))

    async def edge(*args):
        return b"edge-audio"

    monkeypatch.setattr(server, "_get_audio", edge)
    client = TestClient(server.app)
    assert client.get("/health").status_code == 200
    assert client.get("/tts", params={"text": "Hello", "provider": "edge"}).content == b"edge-audio"
