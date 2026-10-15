"""Tail pause correction must retain speech and intentional pauses."""

from types import SimpleNamespace

import numpy as np
import pytest

from zerotts_backend import ZeroTTSBackend


def speech(seconds, sr=48_000):
    return (0.1 * np.sin(2 * np.pi * 220 * np.arange(round(seconds * sr)) / sr)).astype(np.float32)


def test_tail_gap_shortened_without_changing_speech_samples():
    sr = 48_000
    first, last = speech(1), speech(0.3)
    audio = np.concatenate((first, np.zeros(round(sr * 0.21)), last, np.zeros(round(sr * 0.1))))
    fixed = ZeroTTSBackend._shorten_tail_pause(audio, sr, 80)
    assert len(audio) - len(fixed) == round(sr * 0.13)
    np.testing.assert_array_equal(fixed[:len(first)], first)
    np.testing.assert_array_equal(fixed[-round(sr * 0.4):-round(sr * 0.1)], last)


@pytest.mark.parametrize("kind", ["trailing", "short", "middle", "quiet_phone", "disabled"])
def test_other_audio_is_preserved(kind):
    sr = 48_000
    first, last = speech(1), speech(0.3)
    limit = 80
    if kind == "trailing":
        audio = np.r_[first, np.zeros(round(sr * 0.4))]
    elif kind == "short":
        audio = np.r_[first, np.zeros(round(sr * 0.1)), last]
    elif kind == "middle":
        audio = np.r_[first, np.zeros(round(sr * 0.3)), speech(2)]
    elif kind == "quiet_phone":
        audio = np.r_[first, speech(0.3) * 0.02, last]
    else:
        audio = np.r_[first, np.zeros(round(sr * 0.3)), last]
        limit = 0
    np.testing.assert_array_equal(ZeroTTSBackend._shorten_tail_pause(audio, sr, limit), audio)


def test_explicit_final_comma_pause_is_preserved(monkeypatch):
    backend = ZeroTTSBackend()
    monkeypatch.setattr(backend, "_load_engine", lambda: SimpleNamespace(
        sample_rate=48_000, synthesize=lambda *a, **k: speech(1)
    ))
    monkeypatch.setattr(backend, "_shorten_tail_pause", lambda *a: pytest.fail("Punctuation pause must be preserved"))
    assert backend._synthesize_uncached("Gioi lam, hai.", "maichi")[:4] == b"RIFF"
