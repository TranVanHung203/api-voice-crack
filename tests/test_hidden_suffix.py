"""Hidden continuation must preserve text, cut only verified boundaries, and fail safely."""

from types import SimpleNamespace

import numpy as np
import pytest

from zerotts_backend import ZeroTTSBackend
from zerotts_suffix import (
    HiddenSuffixConfig, SuffixBoundary, SuffixDetectionError, WhisperSuffixDetector,
    ctc_token_spans, cut_at_boundary,
)


@pytest.fixture(autouse=True)
def suffix_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", raising=False)
    monkeypatch.delenv("ZEROTTS_HIDDEN_SUFFIX", raising=False)
    monkeypatch.delenv("ZEROTTS_SUFFIX_DEBUG", raising=False)
    monkeypatch.delenv("ZEROTTS_SUFFIX_ASR_MODEL", raising=False)


def waveform(sr=48000):
    audio = np.zeros(round(sr * 1.2), dtype=np.float32)
    for start, stop in [(0.1, 0.62), (0.82, 1.08)]:
        t = np.arange(round(sr * (stop - start))) / sr
        audio[round(start * sr):round(start * sr) + len(t)] = 0.2 * np.sin(2 * np.pi * 200 * t)
    return audio


def boundary():
    return SuffixBoundary(0.8, 0.8, 0.72, [], "Original. Bắt đầu thôi.")


def test_cut_preserves_original_samples_and_removes_suffix():
    original = waveform()
    audio, metadata = cut_at_boundary(original, 48000, boundary(), HiddenSuffixConfig())
    assert 0.770 <= metadata["cut_point"] <= 0.790
    np.testing.assert_array_equal(audio[:round(0.70 * 48000)], original[:round(0.70 * 48000)])
    assert audio[-1] == 0
    assert len(audio) < round(0.8 * 48000)
    assert np.count_nonzero(original[round(0.82 * 48000):]) > 0


def test_cut_refuses_boundary_without_room_after_the_original_word():
    audio = np.full(48000, 0.1, dtype=np.float32)
    unsafe = SuffixBoundary(0.8, 0.8, 0.795, [], "Original. Bắt đầu thôi.")
    with pytest.raises(SuffixDetectionError):
        cut_at_boundary(audio, 48000, unsafe, HiddenSuffixConfig())


def test_fade_changes_only_the_last_15ms():
    audio = np.full(48000, 0.2, dtype=np.float32)
    audio[round(0.72 * 48000):round(0.82 * 48000)] = 0.0001
    result, _ = cut_at_boundary(audio, 48000, boundary(), HiddenSuffixConfig())
    fade = round(0.015 * 48000)
    np.testing.assert_array_equal(result[:-fade], audio[:len(result) - fade])
    assert result[-1] == 0
    assert 0 < result[-fade // 2] < 0.0001


def fake_backend(monkeypatch, detector):
    backend = ZeroTTSBackend()
    backend._suffix_detector = SimpleNamespace(model_name="small", detect=detector)
    calls = []

    def generate(text, **kwargs):
        calls.append((text, kwargs))
        return waveform().reshape(1, -1)

    monkeypatch.setattr(backend, "_load_engine", lambda: SimpleNamespace(sample_rate=48000, synthesize=generate))
    return backend, calls


def test_disabled_never_appends_or_runs_detector(monkeypatch, capsys):
    backend, calls = fake_backend(monkeypatch, lambda *a: pytest.fail("Disabled path must not use ASR"))
    expected = backend._encode_wav(backend._shorten_tail_pause(waveform(), 48000, 80), 48000)
    assert backend._synthesize_uncached("Original.", "hamy") == expected
    assert [c[0] for c in calls] == ["Original."]
    assert capsys.readouterr().out == ""


def test_suffix_uses_one_period_without_space_after_chunking(monkeypatch, capsys):
    backend, calls = fake_backend(monkeypatch, lambda *a: boundary())
    monkeypatch.setenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", "1")
    text = "Original? Second!"
    backend._synthesize_uncached(text, "hamy")
    assert [c[0] for c in calls] == ["Original.Bắt đầu thôi.", "Second.Bắt đầu thôi."]
    assert all(c[1]["min_frames"] == 4 and c[1]["eoa_extra_frames"] == 1 for c in calls)
    assert text == "Original? Second!"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("text,model_text", [
    ("0", "Không, Bắt đầu thôi."), ("không", "Không, Bắt đầu thôi."),
    ("KHÔNG!", "Không, Bắt đầu thôi."),
    ("ba", "Ba, Bắt đầu thôi."), ("BA!", "Ba, Bắt đầu thôi."),
    ("năm", "Năm, Bắt đầu thôi."), ("NĂM?", "Năm, Bắt đầu thôi."),
])
def test_isolated_short_numbers_pause_before_suffix_without_changing_generation(monkeypatch, text, model_text):
    backend, calls = fake_backend(monkeypatch, lambda *a: boundary())
    monkeypatch.setenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", "1")
    backend._synthesize_uncached(text, "hamy")
    assert [c[0] for c in calls] == [model_text]
    assert calls[0][1]["min_frames"] == 4
    assert calls[0][1]["eoa_extra_frames"] == 1
    assert not backend._suffix_fallback_used


@pytest.mark.parametrize("text", ["Một.", "Không, Một, Hai, Ba, Bốn, Năm.", "Có năm.", "Chọn số ba.", "Không có."])
def test_short_word_pause_leaves_other_segments_unchanged(text):
    config = HiddenSuffixConfig()
    assert config.model_text(text) == text.rstrip(".") + ".Bắt đầu thôi."


def test_short_word_pause_respects_custom_suffix_and_disabled_flow(monkeypatch):
    assert HiddenSuffixConfig(suffix=".Tiếp tục.").model_text("Ba.") == "Ba.Tiếp tục."
    backend, calls = fake_backend(monkeypatch, lambda *a: pytest.fail("Disabled path must not use ASR"))
    backend._synthesize_uncached("năm", "hamy")
    assert [c[0] for c in calls] == ["Năm."]


def test_failure_regenerates_legacy_audio_with_same_rng(monkeypatch, caplog):
    backend = ZeroTTSBackend()
    monkeypatch.setattr(backend, "_load_engine", lambda: SimpleNamespace(
        sample_rate=48000, synthesize=lambda *a, **k: np.random.random((1, 4800)).astype(np.float32)
    ))
    original = backend._synthesize_uncached("Original. Second?", "hamy", seed=1234)

    def fail(*args):
        raise SuffixDetectionError("No trailing suffix")

    backend._suffix_detector = SimpleNamespace(model_name="small", detect=fail)
    monkeypatch.setenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", "1")
    before = np.random.get_state()
    result = backend._synthesize_uncached("Original. Second?", "hamy", seed=1234)
    assert result == original
    assert backend._suffix_fallback_used
    assert "using original-text flow" in caplog.text
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_debug_is_opt_in(monkeypatch, capsys):
    backend, _ = fake_backend(monkeypatch, lambda *a: boundary())
    monkeypatch.setenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", "1")
    monkeypatch.setenv("ZEROTTS_SUFFIX_DEBUG", "1")
    backend._synthesize_uncached("Original.", "hamy")
    output = capsys.readouterr().out
    assert "[TTS] Original text: Original." in output
    assert "[TTS] Model text: Original.Bắt đầu thôi." in output
    assert "[TTS] Final cut point:" in output


def test_cache_separates_workaround_and_suffix_settings(monkeypatch, tmp_path):
    backend = ZeroTTSBackend()
    monkeypatch.setattr(backend, "_audio_cache_dir", lambda: tmp_path)
    calls = []
    wav = backend._encode_wav(waveform(), 48000)
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **k: calls.append(k) or wav)
    backend.synthesize_wav("Original.", "hamy")
    monkeypatch.setenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", "1")
    backend.synthesize_wav("Original.", "hamy")
    backend.synthesize_wav("Original.", "hamy")
    monkeypatch.setenv("ZEROTTS_HIDDEN_SUFFIX", "Another suffix.")
    backend.synthesize_wav("Original.", "hamy")
    assert len(calls) == 3
    assert "hidden_suffix_config" not in calls[0]
    assert calls[1]["hidden_suffix_config"]["suffix"] == ".Bắt đầu thôi."
    assert len(list(tmp_path.glob("*.wav"))) == 3


def test_fallback_is_not_cached(monkeypatch, tmp_path):
    backend = ZeroTTSBackend()
    monkeypatch.setenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", "1")
    monkeypatch.setattr(backend, "_audio_cache_dir", lambda: tmp_path)
    calls = []

    def generate(*a, **k):
        backend._suffix_fallback_used = True
        calls.append(k)
        return backend._encode_wav(waveform(), 48000)

    monkeypatch.setattr(backend, "_synthesize_uncached", generate)
    backend.synthesize_wav("Original.", "hamy")
    backend.synthesize_wav("Original.", "hamy")
    assert len(calls) == 2
    assert not backend._audio_cache
    assert not list(tmp_path.glob("*.wav"))


def test_ctc_alignment_requires_blank_between_repeated_characters():
    probabilities = np.array([[0.01, 0.99], [0.99, 0.01], [0.01, 0.99], [0.99, 0.01]])
    spans = ctc_token_spans(np.log(probabilities), [1, 1], blank=0)
    assert spans[0][:2] == (0, 1)
    assert spans[1][:2] == (2, 3)
    with pytest.raises(SuffixDetectionError, match="No complete"):
        ctc_token_spans(np.log(probabilities[:2]), [1, 1], blank=0)


def fake_recognizer(detector, recognized):
    word_objects = [SimpleNamespace(word=w, start=s, end=e, probability=p) for w, s, e, p in recognized]
    segment = SimpleNamespace(text=" ".join(w[0] for w in recognized), words=word_objects)
    detector._model = SimpleNamespace(transcribe=lambda *a, **k: ([segment], None))


def letter_alignment():
    return [
        {"word": "rổ", "start": 0.10, "end": 0.26, "probability": 0.99},
        {"word": "b", "start": 0.42, "end": 0.44, "probability": 0.99},
        {"word": "bắt", "start": 0.80, "end": 0.88, "probability": 0.99},
        {"word": "đầu", "start": 0.90, "end": 0.98, "probability": 0.99},
        {"word": "thôi", "start": 1.00, "end": 1.08, "probability": 0.99},
    ]


def letter_recognizer(detector, final_word=("B", 0.40, 0.73, 0.99)):
    fake_recognizer(detector, [
        ("rổ", 0.10, 0.30, 0.99), final_word,
        ("bắt", 0.76, 0.88, 0.99), ("đầu", 0.90, 0.98, 0.99), ("thôi", 1.00, 1.08, 0.99),
    ])


def test_single_frame_letter_uses_asr_end_to_protect_its_full_spoken_name(monkeypatch):
    detector = WhisperSuffixDetector("small")
    letter_recognizer(detector)
    monkeypatch.setattr(detector, "_align_ctc", lambda *a: letter_alignment())
    point = detector.detect(waveform(), 48000, "rổ B.", HiddenSuffixConfig())
    assert point.prefix_end == 0.73  # The CTC 'b' ends at .44, before the spoken name ends.
    result, metadata = cut_at_boundary(waveform(), 48000, point, HiddenSuffixConfig())
    assert metadata["cut_point"] - .015 > point.prefix_end
    np.testing.assert_array_equal(result[:round(.73 * 48000)], waveform()[:round(.73 * 48000)])


@pytest.mark.parametrize("final_word", [
    ("B", 0.40, 0.40, 0.99), ("B", 0.40, 0.82, 0.99),
])
def test_short_letter_still_fails_without_consistent_independent_timing(monkeypatch, final_word):
    detector = WhisperSuffixDetector("small")
    letter_recognizer(detector, final_word)
    monkeypatch.setattr(detector, "_align_ctc", lambda *a: letter_alignment())
    with pytest.raises(SuffixDetectionError):
        detector.detect(waveform(), 48000, "rổ B.", HiddenSuffixConfig())


def test_confirmed_letter_does_not_regenerate_and_is_cached(monkeypatch, tmp_path):
    detector = WhisperSuffixDetector("small")
    letter_recognizer(detector)
    monkeypatch.setattr(detector, "_align_ctc", lambda *a: letter_alignment())
    backend, calls = fake_backend(monkeypatch, detector.detect)
    monkeypatch.setattr(backend, "_audio_cache_dir", lambda: tmp_path)
    monkeypatch.setenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", "1")
    first = backend.synthesize_wav("rổ B.", "hamy")
    assert backend.synthesize_wav("rổ B.", "hamy") == first
    assert not backend._suffix_fallback_used
    assert [c[0] for c in calls] == ["rổ B.Bắt đầu thôi."]


def test_asr_digit_counting_aligns_the_original_spelled_words(monkeypatch):
    detector = WhisperSuffixDetector("small")
    original = ["một", "hai", "ba", "bốn"]
    suffix = ["bắt", "đầu", "thôi"]
    aligned = [{"word": w, "start": 0.05 + i * 0.15, "end": 0.13 + i * 0.15, "probability": 0.99}
               for i, w in enumerate(original + suffix)]
    fake_recognizer(detector, [(word, interval["start"], interval["end"], 0.99)
                              for word, interval in zip(["1,", "2,", "3,", "4."] + suffix, aligned)])
    transcripts = []
    monkeypatch.setattr(detector, "_align_ctc", lambda audio, transcript, model: transcripts.append(transcript) or aligned)
    point = detector.detect(waveform(), 48000, "một, hai, ba, bốn.", HiddenSuffixConfig())
    assert transcripts == ["một hai ba bốn bắt đầu thôi"]
    assert point.suffix_start == aligned[4]["start"]
    assert point.prefix_end == aligned[3]["end"]
    assert detector.last_diagnostic["recognized_prefix"] == ["1", "2", "3", "4"]
    assert detector.last_diagnostic["validation"] == "suffix_only"


@pytest.mark.parametrize("prefix", [["mục"], ["bảy", "tám", "chín", "mười"], ["đùa", "giấu"]])
def test_original_recognition_does_not_gate_suffix_detection(monkeypatch, prefix):
    detector = WhisperSuffixDetector("small")
    tokens = prefix + ["bắt", "đầu", "thôi"]
    intervals = [{"word": w, "start": .05 + i * .15, "end": .13 + i * .15,
                  "probability": .01 if i < len(prefix) else .99}
                 for i, w in enumerate(tokens)]
    fake_recognizer(detector, [(w["word"], w["start"], w["end"], .99) for w in intervals])
    def align(audio, transcript, model):
        assert transcript == " ".join(tokens)  # Recognized context, not forced original words.
        detector.last_diagnostic["ctc_recognized_text"] = "unrelated original transcription"
        return intervals
    monkeypatch.setattr(detector, "_align_ctc", align)
    point = detector.detect(waveform(), 48000, "một, hai, ba, bốn.", HiddenSuffixConfig())
    assert point.suffix_start == intervals[len(prefix)]["start"]
    assert point.prefix_end == intervals[len(prefix)-1]["end"]
    assert detector.last_diagnostic["validation"] == "suffix_only"
    cut_at_boundary(waveform(), 48000, point, HiddenSuffixConfig())


def mismatched_short_word_detector(monkeypatch, probability=.01):
    detector = WhisperSuffixDetector("small")
    fake_recognizer(detector, [
        ("Mục", .10, .62, .1), ("bắt", .76, .88, .99),
        ("đầu", .90, .98, .99), ("thôi", 1.00, 1.08, .99),
    ])
    def align(audio, transcript, model):
        assert transcript == "mục bắt đầu thôi"
        detector.last_diagnostic["ctc_recognized_text"] = "mục bắt đầu thôi"
        return [
            {"word": "mục", "start": .10, "end": .62, "probability": probability},
            *letter_alignment()[2:],
        ]
    monkeypatch.setattr(detector, "_align_ctc", align)
    return detector


def test_one_misheard_as_muc_is_cut_once_and_cached(monkeypatch, tmp_path):
    detector = mismatched_short_word_detector(monkeypatch)
    point = detector.detect(waveform(), 48000, "Một.", HiddenSuffixConfig())
    assert point.recognized_text.startswith("Mục")
    assert point.prefix_end == .62
    backend, calls = fake_backend(monkeypatch, detector.detect)
    monkeypatch.setattr(backend, "_audio_cache_dir", lambda: tmp_path)
    monkeypatch.setenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", "1")
    first = backend.synthesize_wav("một", "hamy")
    assert backend.synthesize_wav("một", "hamy") == first
    assert [c[0] for c in calls] == ["Một.Bắt đầu thôi."]
    assert not backend._suffix_fallback_used


def test_reported_do_dau_sentence_accepts_misheard_dua_giau(monkeypatch):
    detector = WhisperSuffixDetector("small")
    tokens = ["hãy", "đùa", "giấu", "lớn", "nhé", "bắt", "đầu", "thôi"]
    aligned = [{"word": w, "start": .01+i*.14, "end": .10+i*.14, "probability": .99}
               for i, w in enumerate(tokens)]
    fake_recognizer(detector, [(w["word"], w["start"], w["end"], .99) for w in aligned])
    monkeypatch.setattr(detector, "_align_ctc", lambda *a: aligned)
    point = detector.detect(waveform(), 48000, "Hãy đồ dấu lớn nhé.", HiddenSuffixConfig())
    assert point.suffix_start == aligned[5]["start"]
    result, metadata = cut_at_boundary(waveform(), 48000, point, HiddenSuffixConfig())
    protected = round(point.prefix_end*48000)
    np.testing.assert_array_equal(result[:protected], waveform()[:protected])
    assert metadata["cut_point"] < point.suffix_start


def test_suffix_can_be_located_when_whisper_omits_original_words(monkeypatch):
    detector = WhisperSuffixDetector("small")
    fake_recognizer(detector, [("bắt", .76, .88, .99), ("đầu", .90, .98, .99), ("thôi", 1., 1.08, .99)])
    def align(audio, transcript, model):
        assert transcript == "bắt đầu thôi"
        assert len(audio) == round(1.2*16000) - int(.72*16000)
        return [dict(w, start=w["start"]-.72, end=w["end"]-.72) for w in letter_alignment()[2:]]
    monkeypatch.setattr(detector, "_align_ctc", align)
    point = detector.detect(waveform(), 48000, "Một.", HiddenSuffixConfig())
    assert point.suffix_start == pytest.approx(.8)
    assert point.prefix_end == pytest.approx(.72)
    cut_at_boundary(waveform(), 48000, point, HiddenSuffixConfig())


@pytest.mark.parametrize("field,value", [("probability", .1), ("probability", float("nan")),
                                        ("start", -.1), ("start", .1), ("end", .79)])
def test_unsafe_suffix_alignment_never_produces_a_cut(monkeypatch, field, value):
    detector = mismatched_short_word_detector(monkeypatch)
    align = detector._align_ctc
    def bad_alignment(*args):
        intervals = align(*args)
        intervals[1][field] = value
        return intervals
    monkeypatch.setattr(detector, "_align_ctc", bad_alignment)
    with pytest.raises(SuffixDetectionError):
        detector.detect(waveform(), 48000, "Một.", HiddenSuffixConfig())


def test_asr_must_confirm_the_complete_trailing_suffix(monkeypatch):
    detector = WhisperSuffixDetector("small")
    fake_recognizer(detector, [("Original", 0, 0.4, 1), ("bắt", 0.6, 0.8, 1)])
    monkeypatch.setattr(detector, "_align_ctc", lambda *a: pytest.fail("Missing suffix must fail before CTC"))
    with pytest.raises(SuffixDetectionError, match="complete trailing"):
        detector.detect(waveform(), 48000, "Original.", HiddenSuffixConfig())


def test_repeated_suffix_in_original_uses_the_appended_occurrence(monkeypatch):
    detector = WhisperSuffixDetector("small")
    expected = ["bắt", "đầu", "thôi", "bắt", "đầu", "thôi"]
    aligned = [{"word": w, "start": 0.10 + i * 0.16, "end": 0.18 + i * 0.16, "probability": 0.99}
               for i, w in enumerate(expected)]
    fake_recognizer(detector, [(w["word"], w["start"], w["end"], 0.99) for w in aligned])
    monkeypatch.setattr(detector, "_align_ctc", lambda *a: aligned)
    point = detector.detect(waveform(), 48000, "Bắt đầu thôi.", HiddenSuffixConfig())
    assert point.suffix_start == aligned[3]["start"]
    assert point.prefix_end == aligned[2]["end"]


def test_low_confidence_asr_never_produces_a_cut(monkeypatch):
    detector = WhisperSuffixDetector("small")
    fake_recognizer(detector, [
        ("Original", 0, 0.4, 1), ("bắt", 0.6, 0.8, 0.1),
        ("đầu", 0.8, 1.0, 0.99), ("thôi", 1.0, 1.1, 0.99),
    ])
    monkeypatch.setattr(detector, "_align_ctc", lambda *a: pytest.fail("Low confidence must fail before CTC"))
    with pytest.raises(SuffixDetectionError, match="Low-confidence"):
        detector.detect(waveform(), 48000, "Original.", HiddenSuffixConfig())
