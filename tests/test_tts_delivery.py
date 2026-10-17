"""Audio delivery must reduce transfers without serving stale or truncated audio."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
import tts_delivery as delivery
import zerotts_backend as module
from zerotts_backend import ZeroTTSBackend


@pytest.fixture
def backend(monkeypatch, tmp_path):
    monkeypatch.setattr(ZeroTTSBackend, "_data_dir", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("EDGE_WARMUP", "0")
    monkeypatch.delenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", raising=False)
    monkeypatch.setattr(server, "_load_disk_cache", lambda: None)
    result = ZeroTTSBackend()
    monkeypatch.setattr(server, "zerotts_backend", result)
    return result


PARAMS = {"text": "Cached sentence", "voiceid": "hamy", "provider": "zerotts"}


def test_browser_revalidates_audio_and_changed_settings_replace_old_etag(backend, monkeypatch):
    calls = []
    initial_seed = module._SAMPLING_SEED

    def generate(*a, **k):
        calls.append(1)
        return backend._encode_wav(np.full(4800, .1 if module._SAMPLING_SEED == initial_seed else .2), 48000)

    monkeypatch.setattr(backend, "_synthesize_uncached", generate)
    with TestClient(server.app) as client:
        first = client.get("/tts", params=PARAMS)
        assert first.headers["cache-control"] == "private, no-cache"
        old = first.headers["etag"]
        unchanged = client.get("/tts", params=PARAMS, headers={"If-None-Match": '"other", W/' + old})
        assert unchanged.status_code == 304
        assert unchanged.content == b""
        assert len(calls) == 1
        monkeypatch.setattr(module, "_SAMPLING_SEED", initial_seed + 1)
        changed = client.get("/tts", params=PARAMS, headers={"If-None-Match": old})
        assert changed.status_code == 200
        assert changed.headers["etag"] != old
        assert changed.content != first.content
        assert len(calls) == 2


def test_ranges_and_head_preserve_the_exact_selected_audio(backend, monkeypatch):
    audio = backend._encode_wav(np.arange(4800, dtype=np.float32) / 48000, 48000)
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **k: audio)
    with TestClient(server.app) as client:
        full = client.get("/tts", params=PARAMS)
        part = client.get("/tts", params=PARAMS, headers={"Range": "bytes=10-29", "If-Range": full.headers["etag"]})
        assert part.status_code == 206
        assert part.content == audio[10:30]
        assert part.headers["content-range"] == f"bytes 10-29/{len(audio)}"
        tail = client.get("/tts", params=PARAMS, headers={"Range": "bytes=-17"})
        assert tail.content == audio[-17:]
        unsatisfied = client.get("/tts", params=PARAMS, headers={"Range": "bytes=999999-"})
        assert unsatisfied.status_code == 416
        assert unsatisfied.headers["content-range"] == f"bytes */{len(audio)}"
        stale = client.get("/tts", params=PARAMS, headers={"Range": "bytes=0-19", "If-Range": '"stale"'})
        assert stale.status_code == 200 and stale.content == audio
        head = client.head("/tts", params=PARAMS)
        assert head.status_code == 200 and head.content == b""
        assert int(head.headers["content-length"]) == len(audio)
        assert head.headers["etag"] == full.headers["etag"]
        cors = client.get("/tts", params=PARAMS, headers={"Origin": "http://localhost:5000"})
        assert "ETag" in cors.headers["access-control-expose-headers"]


@pytest.mark.parametrize("storage", ["RAM", "disk"])
def test_cached_api_response_does_not_wait_for_another_inference(backend, monkeypatch, storage):
    audio = backend._encode_wav(np.zeros(4800), 48000)
    entered, release = threading.Event(), threading.Event()

    def generate(text, *a, **k):
        if text == "slow":
            entered.set()
            assert release.wait(5), "Test did not release inference"
        return audio

    monkeypatch.setattr(backend, "_synthesize_uncached", generate)
    backend.synthesize_wav(PARAMS["text"], "hamy")
    if storage == "disk":
        backend._audio_cache.clear()
        backend._audio_cache_bytes = 0
    with TestClient(server.app) as client, ThreadPoolExecutor(2) as pool:
        slow = pool.submit(client.get, "/tts", params=PARAMS | {"text": "slow"})
        try:
            assert entered.wait(2)
            hit = pool.submit(client.get, "/tts", params=PARAMS)
            response = hit.result(timeout=1)
            assert response.status_code == 200 and response.content == audio
            assert not slow.done()
        finally:
            release.set()
        assert slow.result(timeout=2).status_code == 200


@pytest.mark.parametrize("path,method", [("/tts", "get"), ("/tts", "post"), ("/zerotts/synthesize", "post")])
def test_mp3_opt_in_retains_wav_as_the_default(backend, monkeypatch, path, method):
    audio = backend._encode_wav(np.zeros(4800), 48000)
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **k: audio)
    monkeypatch.setattr(server, "encode_mp3", lambda wav, directory: b"ID3-compressed")
    with TestClient(server.app) as client:
        params = PARAMS | {"format": "mp3", "voice_id": "hamy"}
        response = client.request(method, path, **({"params": params} if method == "get" else {"json": params}))
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == b"ID3-compressed"
        response = client.get("/tts", params=PARAMS)
        assert response.headers["content-type"] == "audio/wav"
        assert response.content == audio


def test_mp3_cache_reuses_encoding_and_invalidates_when_audio_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(delivery, "_MP3_CACHE", OrderedDict())
    calls = []

    def encode(wav):
        calls.append(1)
        return b"ID3" + wav

    monkeypatch.setattr(delivery, "_encode_mp3_data", encode)
    assert delivery.encode_mp3(b"wave-one", tmp_path) == b"ID3wave-one"
    assert delivery.encode_mp3(b"wave-one", tmp_path) == b"ID3wave-one"
    delivery._MP3_CACHE.clear()
    assert delivery.encode_mp3(b"wave-one", tmp_path) == b"ID3wave-one"
    assert len(calls) == 1
    assert delivery.encode_mp3(b"wave-two", tmp_path) == b"ID3wave-two"
    assert len(calls) == 2


def test_missing_encoder_returns_a_clear_error(backend, monkeypatch):
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **k: backend._encode_wav(np.zeros(100), 48000))
    monkeypatch.setattr(server, "encode_mp3", lambda *a: (_ for _ in ()).throw(delivery.AudioEncodingError("ffmpeg missing")))
    with TestClient(server.app) as client:
        response = client.get("/tts", params=PARAMS | {"format": "mp3"})
        assert response.status_code == 503
        assert "ffmpeg" in response.json()["detail"]
