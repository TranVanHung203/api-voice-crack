"""Persistent audio reuse, invalidation and concurrent request behavior."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import zerotts_backend as module
from zerotts_backend import ZeroTTSBackend, ZeroTTSBackendError


@pytest.fixture
def backend(monkeypatch, tmp_path):
    monkeypatch.setattr(ZeroTTSBackend, "_data_dir", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(ZeroTTSBackend, "_bundled_voices_dir", staticmethod(lambda: tmp_path / "bundled"))
    monkeypatch.setattr(ZeroTTSBackend, "_cache_dir", classmethod(lambda cls: tmp_path / "model_cache"))
    monkeypatch.delenv("ZEROTTS_MODEL_DIR", raising=False)
    return ZeroTTSBackend()


def test_ram_and_disk_hits_skip_model_loading(backend, monkeypatch):
    calls = []
    wav = backend._encode_wav(np.ones(4800, dtype=np.float32) * 0.1, 48_000)

    def generate(*args, **options):
        calls.append(options)
        return wav

    monkeypatch.setattr(backend, "_synthesize_uncached", generate)
    assert backend.synthesize_wav("Xin chao", "hamy") == wav
    assert backend.synthesize_wav(" Xin chao ", "hamy") == wav
    assert len(calls) == 1
    restarted = ZeroTTSBackend()
    monkeypatch.setattr(restarted, "_load_engine", lambda: pytest.fail("Disk hit must not load model"))
    assert restarted.synthesize_wav("Xin chao", "hamy") == wav


@pytest.mark.parametrize("change", [
    {"text": "Khac"}, {"voice_id": "maichi"}, {"cfg_scale": 1.1},
    {"audio_temperature": 0.7}, {"audio_repetition_penalty": 1.3},
    {"max_frames": 500}, {"eoa_extra_frames": 4},
])
def test_generation_settings_invalidate_cache(backend, monkeypatch, change):
    calls = []
    wav = backend._encode_wav(np.zeros(4800), 48_000)

    def generate(*args, **options):
        calls.append(options)
        return wav

    monkeypatch.setattr(backend, "_synthesize_uncached", generate)
    backend.synthesize_wav("Xin chao", "hamy")
    backend.synthesize_wav(**({"text": "Xin chao", "voice_id": "hamy"} | change))
    assert len(calls) == 2


def test_chunk_setting_invalidates_audio_cache(backend, monkeypatch):
    calls = []
    wav = backend._encode_wav(np.zeros(4800), 48_000)
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **k: calls.append(k) or wav)
    backend.synthesize_wav("Xin chao", "hamy", max_chunk_sec=1)
    backend.synthesize_wav("Xin chao", "hamy", max_chunk_sec=15)
    assert len(calls) == 2
    assert [options["max_chunk_sec"] for options in calls] == [1.0, 15.0]


def test_sampling_constants_and_voice_changes_invalidate_cache(backend, monkeypatch):
    calls = []
    wav = backend._encode_wav(np.zeros(4800), 48_000)
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **k: calls.append(k) or wav)
    backend.synthesize_wav("Xin chao", "hamy")
    monkeypatch.setattr(module, "_MIN_FRAMES", module._MIN_FRAMES + 1)
    backend.synthesize_wav("Xin chao", "hamy")
    monkeypatch.setattr(module, "_TEXT_TEMPERATURE", 0.7)
    backend.synthesize_wav("Xin chao", "hamy")
    monkeypatch.setattr(module, "_SAMPLING_SEED", module._SAMPLING_SEED + 1)
    backend.synthesize_wav("Xin chao", "hamy")
    voice_dir = backend._voices_dir() / "hamy"
    voice_dir.mkdir(parents=True)
    (voice_dir / "voice.npz").write_bytes(b"new voice")
    backend.synthesize_wav("Xin chao", "hamy")
    assert len(calls) == 5


def test_concurrent_identical_requests_generate_once(backend, monkeypatch):
    calls = []
    wav = backend._encode_wav(np.zeros(4800), 48_000)
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **k: calls.append(k) or wav)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: backend.synthesize_wav("Xin chao", "hamy"), range(4)))
    assert results == [wav] * 4
    assert len(calls) == 1


def test_failed_synthesis_is_not_cached(backend, monkeypatch):
    def fail(*args, **kwargs):
        raise ZeroTTSBackendError("failed")

    monkeypatch.setattr(backend, "_synthesize_uncached", fail)
    with pytest.raises(ZeroTTSBackendError):
        backend.synthesize_wav("Xin chao", "hamy")
    assert not backend._audio_cache
    assert not list(backend._audio_cache_dir().glob("*.wav"))
