"""Optional hidden-continuation trimming. No models are imported while disabled."""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
SUFFIX_CACHE_VERSION = 7
DEFAULT_HIDDEN_SUFFIX = ".Bắt đầu thôi."
_SHORT_WORD_PAUSE = frozenset({"không", "ba", "năm"})


class SuffixDetectionError(RuntimeError):
    """A suffix boundary could not be established safely."""


def words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold(), re.UNICODE)


def alignment_words(tokens: list[str]) -> list[str]:
    """Expand ASR integers for the character-level alignment transcript.

    Whisper may write a spoken counting list as ``1, 2, 3, 4``. Expand each
    already-tokenized integer independently so commas cannot become decimals.
    Use the same number pronunciation rules as the existing TTS normalizer.
    """
    if not any(re.fullmatch(r"[0-9]+", token) for token in tokens):
        return tokens
    from zerotts.text_norm import expand_number

    return [word for token in tokens for word in (
        words(expand_number(token)) if re.fullmatch(r"[0-9]+", token) else [token]
    )]


@dataclass(frozen=True)
class HiddenSuffixConfig:
    enabled: bool = False
    suffix: str = DEFAULT_HIDDEN_SUFFIX
    asr_model: str = "small"
    alignment_model: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    debug: bool = False
    margin_ms: int = 20
    fade_ms: int = 15
    agreement_ms: int = 40
    min_probability: float = 0.35

    @classmethod
    def from_env(cls) -> HiddenSuffixConfig:
        flag = lambda name: os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}
        return cls(
            enabled=flag("ZEROTTS_ENABLE_HIDDEN_SUFFIX"),
            # Leading whitespace is intentional model context, appended after chunking.
            suffix=os.getenv("ZEROTTS_HIDDEN_SUFFIX", DEFAULT_HIDDEN_SUFFIX).rstrip(),
            asr_model=os.getenv("ZEROTTS_SUFFIX_ASR_MODEL", "small"),
            alignment_model=os.getenv("ZEROTTS_SUFFIX_ALIGN_MODEL", "nguyenvulebinh/wav2vec2-base-vietnamese-250h"),
            debug=flag("ZEROTTS_SUFFIX_DEBUG"),
        )

    def cache_options(self) -> dict:
        result = asdict(self)
        result.pop("debug")
        result["version"] = SUFFIX_CACHE_VERSION
        result["model_files"] = {}
        for model in [self.asr_model, self.alignment_model]:
            model_path = Path(model)
            if model_path.is_dir():
                result["model_files"][model] = [
                    (p.name, p.stat().st_size, p.stat().st_mtime_ns)
                    for p in sorted(model_path.iterdir()) if p.is_file()
                ]
        return result

    def model_text(self, original: str) -> str:
        """Use a leading comma or period as the internal boundary, after chunking."""
        tokens = words(original)
        if self.suffix == DEFAULT_HIDDEN_SUFFIX and len(tokens) == 1 and tokens[0] in _SHORT_WORD_PAUSE:
            # These isolated numbers need a phrase pause before the continuation.
            # Apply after chunking; the user/API text and custom suffixes stay intact.
            return original.rstrip(".!?…,:; \t\r\n") + ", " + self.suffix.lstrip(".")
        if self.suffix.startswith((",", ".")):
            # Keep one boundary mark, rather than producing 'Ba..' or 'Ba.,'.
            # The original request/alignment text is kept intact.
            return original.rstrip(".!?…,:; \t\r\n") + self.suffix
        return original + self.suffix

    @classmethod
    def from_cache_options(cls, options: dict) -> HiddenSuffixConfig:
        values = {k: v for k, v in options.items() if k not in {"version", "model_files"}}
        return cls(**values, debug=cls.from_env().debug)


@dataclass
class SuffixBoundary:
    suffix_start: float
    asr_start: float
    prefix_end: float
    aligned_words: list[dict]
    recognized_text: str


def ctc_token_spans(log_probabilities, tokens: list[int], blank: int):
    """CTC Viterbi alignment, including mandatory blanks between repeated labels.

    Returns token spans in emission frames and mean acoustic probabilities.
    Unlike a free ASR decode, this aligns the complete known transcript.
    """
    import numpy as np

    emissions = np.asarray(log_probabilities)
    frames = emissions.shape[0]
    if not tokens or frames < len(tokens):
        raise SuffixDetectionError("Transcript is too long for the CTC emissions")
    labels = np.full(2 * len(tokens) + 1, blank, dtype=np.int64)
    labels[1::2] = tokens
    previous = np.full(len(labels), -np.inf)
    previous[0], previous[1] = emissions[0, blank], emissions[0, tokens[0]]
    back = np.zeros((frames, len(labels)), dtype=np.uint8)
    can_skip = np.zeros(len(labels), dtype=bool)
    can_skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    for t in range(1, frames):
        step = np.r_[-np.inf, previous[:-1]]
        skip = np.r_[-np.inf, -np.inf, previous[:-2]]
        skip[~can_skip] = -np.inf
        scores = np.stack([previous, step, skip])
        choice = scores.argmax(axis=0)
        back[t] = choice
        previous = scores[choice, np.arange(len(labels))] + emissions[t, labels]
    state = len(labels) - 1 if previous[-1] > previous[-2] else len(labels) - 2
    if not np.isfinite(previous[state]):
        raise SuffixDetectionError("No complete CTC alignment path")
    assignments = [[] for _ in tokens]
    for t in range(frames - 1, -1, -1):
        if state % 2:
            assignments[state // 2].append(t)
        if t:
            state -= int(back[t, state])
    if any(not span for span in assignments):
        raise SuffixDetectionError("CTC alignment omitted a transcript token")
    return [
        (min(span), max(span) + 1, float(np.exp(emissions[span, token]).mean()))
        for span, token in zip(assignments, tokens)
    ]


class WhisperSuffixDetector:
    """Verify the spoken suffix, then align the known transcript using CTC.

    Uses existing faster-whisper for suffix verification and a Vietnamese CTC
    model for character alignment. An alignment alone is insufficient: it can
    force timestamps onto words that were never spoken.
    """

    def __init__(self, model: str):
        self.model_name = model
        self._model: Any = None
        self._lock = threading.Lock()
        self.last_diagnostic: dict = {}
        self._alignment_model_name = None
        self._alignment_model = None
        self._processor = None

    def _align_ctc(self, resampled, transcript: str, model_name: str):
        import numpy as np
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        if self._alignment_model_name != model_name:
            self._processor = Wav2Vec2Processor.from_pretrained(model_name)
            self._alignment_model = Wav2Vec2ForCTC.from_pretrained(model_name).eval()
            self._alignment_model_name = model_name
        tokens = self._processor.tokenizer(transcript, add_special_tokens=False).input_ids
        if self._processor.tokenizer.unk_token_id in tokens:
            raise SuffixDetectionError("The alignment model cannot encode this transcript")
        inputs = self._processor(resampled, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = self._alignment_model(**inputs).logits[0]
            emissions = torch.log_softmax(logits, dim=-1).cpu().numpy()
        # Diagnostic only: recognition of the original text is not a condition
        # for trimming the independently verified hidden suffix.
        self.last_diagnostic["ctc_recognized_text"] = self._processor.tokenizer.decode(
            np.argmax(emissions, axis=-1)
        )
        spans = ctc_token_spans(emissions, tokens, self._alignment_model.config.pad_token_id)
        # Character-token contract is checked, rather than silently assuming
        # that an arbitrary configurable model has this tokenizer vocabulary.
        encoded_chars = self._processor.tokenizer.convert_ids_to_tokens(tokens)
        delimiter = self._processor.tokenizer.word_delimiter_token
        if "".join(encoded_chars).replace(delimiter, " ") != transcript:
            raise SuffixDetectionError("Expected a character-level Vietnamese CTC tokenizer")
        seconds_per_frame = len(resampled) / 16000 / len(emissions)
        output, group = [], []
        for char, span in zip(encoded_chars + [delimiter], spans + [(0, 0, 0)]):
            if char == delimiter:
                if group:
                    output.append({
                        "word": "".join(c for c, _ in group),
                        "start": group[0][1][0] * seconds_per_frame,
                        "end": group[-1][1][1] * seconds_per_frame,
                        "probability": float(np.mean([s[2] for _, s in group])),
                    })
                    group = []
            else:
                group.append((char, span))
        return output

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise SuffixDetectionError(
                    "Hidden suffix requires faster-whisper: install requirements-hidden-suffix.txt "
                    f"in {os.sys.executable}"
                ) from exc
            self._model = WhisperModel(
                self.model_name, device="cpu", compute_type="int8", cpu_threads=4,
            )
        return self._model

    def detect(self, audio, sample_rate: int, original: str, config: HiddenSuffixConfig) -> SuffixBoundary:
        import numpy as np
        from scipy.signal import resample_poly

        source_words, suffix_words = words(original), words(config.suffix)
        if not source_words or not suffix_words:
            raise SuffixDetectionError("Original text and hidden suffix must contain words")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate <= 0 or not samples.size or not np.isfinite(samples).all():
            raise SuffixDetectionError("Invalid waveform")
        divisor = math.gcd(sample_rate, 16000)
        resampled = resample_poly(samples, 16000 // divisor, sample_rate // divisor).astype(np.float32)
        if len(resampled) > 29 * 16000:
            raise SuffixDetectionError("Audio exceeds the 29-second alignment window")

        with self._lock:
            self.last_diagnostic = {}
            model = self._load()
            segments, _ = model.transcribe(
                resampled, language="vi", beam_size=5, temperature=0,
                condition_on_previous_text=False, vad_filter=False, word_timestamps=True,
            )
            segments = list(segments)
            recognized = "".join(s.text for s in segments)
            self.last_diagnostic["recognized_text"] = recognized
            recognized_words = []
            for segment in segments:
                for word in segment.words or []:
                    for token in words(word.word):
                        recognized_words.append((token, float(word.start), float(word.end), float(word.probability)))
            count = len(suffix_words)
            if len(recognized_words) < count or [w[0] for w in recognized_words[-count:]] != suffix_words:
                raise SuffixDetectionError(f"ASR did not confirm the complete trailing suffix: {recognized!r}")
            tail = recognized_words[-count:]
            self.last_diagnostic["asr_suffix_words"] = tail
            duration = len(samples) / sample_rate
            if any(
                not all(math.isfinite(v) for v in w[1:])
                or not 0 <= w[1] < w[2] <= duration
                or not config.min_probability <= w[3] <= 1
                for w in tail
            ) or any(a[2] > b[1] for a, b in zip(tail, tail[1:])):
                raise SuffixDetectionError("Low-confidence or zero-duration suffix words")
            recognized_prefix = [w[0] for w in recognized_words[:-count]]
            context_words = alignment_words(recognized_prefix)
            self.last_diagnostic.update(
                validation="suffix_only", recognized_prefix=recognized_prefix,
                alignment_context=context_words,
            )

            # Spaces here belong only to the alignment transcript. The TTS model
            # receives the configured boundary punctuation and hidden suffix.
            # The prefix is acoustic context, not a comparison with the user's
            # text. Whisper spelling mistakes must not reject a correct suffix.
            transcript = " ".join(context_words + suffix_words)
            offset_samples = 0
            if not context_words:
                # Whisper can omit all original words. Restrict suffix-only
                # alignment to its ASR window rather than forcing 'b' onto the
                # beginning of the original speech.
                offset_samples = max(0, int((tail[0][1] - config.agreement_ms / 1000) * 16000))
            aligned = self._align_ctc(resampled[offset_samples:], transcript, config.alignment_model)
            offset = offset_samples / 16000
            aligned = [dict(w, start=float(w["start"]) + offset, end=float(w["end"]) + offset)
                       for w in aligned]
            self.last_diagnostic["aligned_words"] = [
                {"word": w["word"], "start": float(w["start"]), "end": float(w["end"]),
                 "probability": float(w["probability"])} for w in aligned
            ]
            aligned_tokens = [token for word in aligned for token in words(word["word"])]
            if aligned_tokens != context_words + suffix_words or len(aligned) != len(aligned_tokens):
                raise SuffixDetectionError("Alignment did not produce one interval per expected word")
            boundary_index = len(context_words)
            suffix_start = float(aligned[boundary_index]["start"])
            prefix_end = float(aligned[boundary_index - 1]["end"]) if boundary_index else offset
            tolerance = config.agreement_ms / 1000
            if not tail[0][1] - tolerance <= suffix_start <= tail[0][2] + tolerance:
                raise SuffixDetectionError(f"CTC suffix onset {suffix_start:.3f}s is outside the ASR word interval")
            if not 0 < suffix_start < len(samples) / sample_rate:
                raise SuffixDetectionError("Suffix onset is outside the waveform")
            suffix_alignment = aligned[boundary_index:]
            if any(
                not all(math.isfinite(float(w[k])) for k in ("start", "end", "probability"))
                or not 0 <= float(w["start"]) < float(w["end"]) <= duration
                or not config.min_probability <= float(w["probability"]) <= 1
                for w in suffix_alignment
            ):
                raise SuffixDetectionError("Low-confidence forced alignment of the suffix")
            if boundary_index:
                # Protect the preceding word's timing regardless of its spelling
                # or recognition score. A one-frame CTC character (e.g. B) is
                # not the complete spoken vowel: retain the ASR end as well.
                last_asr = recognized_words[-count - 1]
                if not all(math.isfinite(v) for v in last_asr[1:3]) or not 0 <= last_asr[1] < last_asr[2] <= duration:
                    raise SuffixDetectionError("Invalid pre-suffix word timing")
                prefix_end = max(prefix_end, last_asr[2])
                self.last_diagnostic["asr_final_word"] = last_asr
            if not math.isfinite(prefix_end) or not 0 < prefix_end < suffix_start:
                raise SuffixDetectionError("No safe boundary after the pre-suffix speech")
            intervals = [
                {"word": w["word"], "start": float(w["start"]), "end": float(w["end"]),
                 "probability": float(w["probability"])} for w in aligned
            ]
            return SuffixBoundary(suffix_start, tail[0][1], prefix_end, intervals, recognized)


def cut_at_boundary(audio, sample_rate: int, boundary: SuffixBoundary, config: HiddenSuffixConfig):
    """Refine a lexical timestamp within a bounded low-energy boundary region."""
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    onset = boundary.suffix_start
    # CTC locates the first character of the suffix; ASR word timestamps can
    # include the preceding pause. Protect the last aligned original character.
    protected_end = boundary.prefix_end + config.fade_ms / 1000 + 0.001
    left = max(0, round(max(onset - 0.060, protected_end) * sample_rate))
    right = min(len(samples), round((onset - 0.010) * sample_rate))
    target = round((onset - config.margin_ms / 1000) * sample_rate)
    window = max(1, round(0.005 * sample_rate))
    if right - left < window or target <= 0:
        raise SuffixDetectionError("No usable pre-suffix cut interval")
    local = samples[max(0, left - sample_rate // 3):min(len(samples), right + sample_rate // 10)]
    reference = float(np.sqrt(np.mean(local.astype(np.float64) ** 2)))
    candidates = []
    for i in range(left + window // 2, right - window // 2 + 1, max(1, window // 2)):
        region = samples[i - window // 2:i + window // 2]
        energy = float(np.sqrt(np.mean(region.astype(np.float64) ** 2)))
        if energy <= max(0.0003, reference * 0.10):
            candidates.append((abs(i - target), i))
    method = "low_energy_zero_crossing"
    if candidates:
        cut = min(candidates)[1]
    else:
        # Speech can be coarticulated with no silent gap. Use the validated CTC
        # boundary only if the whole fade still lies after the original word.
        cut = max(left + window // 2, min(target, right - window // 2))
        if cut - round(config.fade_ms / 1000 * sample_rate) < round(boundary.prefix_end * sample_rate):
            raise SuffixDetectionError("No safe boundary after the final original word")
        method = "ctc_timestamp_zero_crossing"
    # Snap at most 1 ms to a zero crossing inside the accepted quiet region.
    radius = max(1, round(0.001 * sample_rate))
    lo, hi = max(left, cut - radius), min(right, cut + radius)
    crossings = np.flatnonzero(np.signbit(samples[lo:hi - 1]) != np.signbit(samples[lo + 1:hi])) + lo + 1
    if crossings.size:
        cut = int(crossings[np.argmin(np.abs(crossings - cut))])
    if cut <= 0 or cut >= len(samples):
        raise SuffixDetectionError("Invalid final cut")
    result = samples[:cut].copy()
    fade = min(len(result), round(config.fade_ms / 1000 * sample_rate))
    if fade:
        result[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return result, {"suffix_start": boundary.suffix_start, "asr_start": boundary.asr_start,
                    "protected_prefix_end": boundary.prefix_end,
                    "cut_point": cut / sample_rate, "before_duration": len(samples) / sample_rate,
                    "after_duration": len(result) / sample_rate,
                    "recognized_text": boundary.recognized_text, "aligned_words": boundary.aligned_words,
                    "cut_method": method}
