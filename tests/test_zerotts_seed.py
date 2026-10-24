"""A new sampling seed must reach ZeroTTS and produce a separate cached result."""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from zerotts_backend import ZeroTTSBackend


@pytest.mark.parametrize("endpoint", ["get", "post", "zerotts"])
@pytest.mark.parametrize("seed", [0, 7])
def test_seed_reaches_backend(monkeypatch, endpoint, seed):
    captured = []
    monkeypatch.setattr(server, "FORCE_VOICE_ID", False)

    def generate(text, voice_id, **options):
        captured.append(options["seed"])
        return b"RIFF-seeded"

    monkeypatch.setattr(server.zerotts_backend, "synthesize_wav", generate)
    client = TestClient(server.app)
    payload = {"text": "Xin chao", "voiceid": "test", "provider": "zerotts", "seed": seed}
    if endpoint == "get":
        response = client.get("/tts", params=payload)
    elif endpoint == "post":
        response = client.post("/tts", json=payload)
    else:
        response = client.post("/zerotts/synthesize", json={"text": "Xin chao", "voice_id": "test", "seed": seed})
    assert response.status_code == 200
    assert captured == [seed]


@pytest.mark.parametrize("seed", [-1, 4294967296])
def test_api_rejects_out_of_range_seed(seed):
    client = TestClient(server.app)
    assert client.get("/tts", params={"text": "Hello", "seed": seed}).status_code == 422
    assert client.post("/tts", json={"text": "Hello", "seed": seed}).status_code == 422
    assert client.post("/zerotts/synthesize", json={"text": "Hello", "voice_id": "test", "seed": seed}).status_code == 422


def test_seeds_generate_distinct_cached_audio_and_restore_rng(monkeypatch, tmp_path):
    backend = ZeroTTSBackend()
    monkeypatch.setattr(backend, "_audio_cache_dir", lambda: tmp_path)
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs)
        return np.random.random((1, 4800)).astype(np.float32)

    monkeypatch.setattr(backend, "_load_engine", lambda: SimpleNamespace(sample_rate=48_000, synthesize=generate))
    before = np.random.get_state()
    first = backend.synthesize_wav("Hello.", "hamy", seed=0)
    second = backend.synthesize_wav("Hello.", "hamy", seed=7)
    assert first != second
    assert first == backend.synthesize_wav("Hello.", "hamy", seed=0)
    assert second == backend.synthesize_wav("Hello.", "hamy", seed=7)
    assert len(calls) == 2
    assert len(list(tmp_path.glob("*.wav"))) == 2
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("text,seed", [("Năm", 11), ("Trên tờ lịch này có", 7), ("Đây là gang tay", 7)])
def test_prompt_default_matches_calibrated_seed_in_real_generation_path(monkeypatch, text, seed):
    monkeypatch.delenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", raising=False)
    backend = ZeroTTSBackend()
    monkeypatch.setattr(backend, "_load_engine", lambda: SimpleNamespace(
        sample_rate=48_000,
        synthesize=lambda *a, **k: np.random.random((1, 4800)).astype(np.float32),
    ))
    voice = "VpYLlCQEgcoHgycJstNP"
    before = np.random.get_state()
    tuned = backend._synthesize_uncached(text, voice)
    assert tuned == backend._synthesize_uncached(text, voice, seed=seed)
    assert tuned != backend._synthesize_uncached(text, voice, seed=23011110)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
