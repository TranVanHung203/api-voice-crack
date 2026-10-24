"""One prompt's slower delivery must reuse source speech and preserve pitch."""

from io import BytesIO
import os
import shutil
import unicodedata
import wave

import numpy as np
import pytest

import tts_delivery
import zerotts_backend as settings
from zerotts_backend import ZeroTTSBackend, ZeroTTSBackendError
from zerotts_suffix import HiddenSuffixConfig

VOICE = "VpYLlCQEgcoHgycJstNP"


@pytest.fixture
def backend(monkeypatch, tmp_path):
    monkeypatch.setattr(ZeroTTSBackend, "_audio_cache_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(ZeroTTSBackend, "_bundled_voices_dir", staticmethod(lambda: tmp_path / "bundled"))
    monkeypatch.setenv("ZEROTTS_DATA_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("HF_AUDIO_CACHE_REPO", raising=False)
    return ZeroTTSBackend()


@pytest.mark.parametrize("text,voice,slowed", [
    ("Đây là gang tay", VOICE, True),
    ("  ĐÂY LÀ GANG TAY!  ", VOICE, True),
    (unicodedata.normalize("NFD", "Đây là gang tay."), VOICE, True),
    ("Đây là gang tay", "hamy", False),
    ("Đây là gang tay của bạn.", VOICE, False),
    ("Quan sát một gang tay.", VOICE, False),
    ("Đây là\ngang tay", VOICE, False),
])
def test_slowing_is_limited_to_the_requested_phrase(backend, monkeypatch, text, voice, slowed):
    source = backend._encode_wav(np.zeros(24000), 24000)
    adjusted = backend._encode_wav(np.zeros(26400), 24000)
    calls = []
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **k: source)
    monkeypatch.setattr(tts_delivery, "change_wav_tempo", lambda wav, tempo, directory: calls.append((wav, tempo)) or adjusted)
    assert backend.synthesize_wav(text, voice) == (adjusted if slowed else source)
    assert calls == ([(source, 1.0)] if slowed else [])


@pytest.mark.skipif(not (os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")), reason="ffmpeg required")
def test_real_tempo_preserves_pitch_and_reuses_source_after_restart(backend, monkeypatch, tmp_path):
    sr = 24000
    source = backend._encode_wav(.2 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr), sr)
    generations = []
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **k: generations.append(True) or source)
    slowed = backend.synthesize_wav("Đây là gang tay.", VOICE)
    with wave.open(BytesIO(slowed), "rb") as wav:
        assert wav.getframerate() == sr
        signal = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float)
    assert len(signal) / sr == pytest.approx(1.0, abs=.03)
    spectrum = abs(np.fft.rfft(signal * np.hanning(len(signal))))
    assert np.fft.rfftfreq(len(signal), 1 / sr)[spectrum.argmax()] == pytest.approx(220, abs=1)
    assert len(generations) == 1
    assert next(tmp_path.glob("*.wav")).read_bytes() == source
    assert slowed == source  # 100% tempo is a byte-preserving pass-through.
    monkeypatch.setattr(tts_delivery.subprocess, "run", lambda *a, **k: pytest.fail("Must reuse the processed WAV"))
    assert backend.synthesize_wav("Đây là gang tay.", VOICE) == slowed
    restarted = ZeroTTSBackend()
    monkeypatch.setattr(restarted, "_load_engine", lambda: pytest.fail("Disk cache must skip inference"))
    assert restarted.synthesize_wav("Đây là gang tay.", VOICE) == slowed


def test_tempo_failure_keeps_cached_source_for_retry(backend, monkeypatch, tmp_path):
    source = backend._encode_wav(np.zeros(24000), 24000)
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **k: source)

    def fail(*args):
        raise tts_delivery.AudioEncodingError("ffmpeg unavailable")

    monkeypatch.setattr(tts_delivery, "change_wav_tempo", fail)
    with pytest.raises(ZeroTTSBackendError, match="FFmpeg"):
        backend.synthesize_wav("Đây là gang tay", VOICE)
    assert next(tmp_path.glob("*.wav")).read_bytes() == source


def test_new_handspan_reading_does_not_reuse_the_clipped_source(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", "1")
    text = "Đây là gang tay."
    old_options = dict(cfg_scale=1., audio_temperature=.8, audio_repetition_penalty=1.2,
        max_chunk_sec=15., max_frames=1500, eoa_extra_frames=1, seed=7,
        hidden_suffix_config=HiddenSuffixConfig(enabled=True, suffix="; Bắt đầu thôi.").cache_options())
    clipped = backend._encode_wav(np.zeros(24000), 24000)
    monkeypatch.setattr(settings, "_HANDSPAN_ENDING_VERSION", 0)
    old_key = backend._audio_cache_key(text, VOICE, old_options)
    monkeypatch.setattr(settings, "_HANDSPAN_ENDING_VERSION", 1)
    (tmp_path / f"{old_key}.wav").write_bytes(clipped)
    complete = backend._encode_wav(np.ones(24000) * .1, 24000)
    slowed = backend._encode_wav(np.ones(30000) * .1, 24000)
    generations, processing = [], []
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **k: generations.append(k) or complete)
    monkeypatch.setattr(tts_delivery, "change_wav_tempo", lambda wav, tempo, directory: processing.append((wav, tempo)) or slowed)
    assert backend.synthesize_wav(text, VOICE) == slowed
    assert generations[0]["seed"] == 7
    assert generations[0]["hidden_suffix_config"]["suffix"] == "; Bắt đầu thôi."
    assert processing == [(complete, 1.0)]
    assert backend.synthesize_wav(text, VOICE) == slowed
    assert len(generations) == 1


@pytest.mark.skipif(not (os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")), reason="ffmpeg required")
def test_real_word_extension_preserves_prefix_and_pitch(backend):
    sr = 24000
    original = (.2 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
    extended = backend._lengthen_final_word(original, sr, .6)
    np.testing.assert_array_equal(extended[:round(.59 * sr)], original[:round(.59 * sr)])
    assert len(extended) / sr == pytest.approx(.6 + .4 / .75, abs=.03)
    spectrum = abs(np.fft.rfft(extended * np.hanning(len(extended))))
    assert np.fft.rfftfreq(len(extended), 1 / sr)[spectrum.argmax()] == pytest.approx(220, abs=1)
