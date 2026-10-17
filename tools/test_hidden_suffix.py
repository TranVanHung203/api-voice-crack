"""Generate before/after WAVs through the same optional backend path as the API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zerotts_backend import ZeroTTSBackend, _SAMPLING_SEED
from zerotts_suffix import HiddenSuffixConfig, WhisperSuffixDetector, cut_at_boundary


SENTENCES = [
    "Trong bể có bao nhiêu con cá?",
    "Hãy cùng khám phá nhé!",
    "Có tất cả bao nhiêu quả cam?",
    "Con hãy chọn đáp án đúng nhé!",
    "Bạn nhìn thấy bao nhiêu con vịt?",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="VpYLlCQEgcoHgycJstNP")
    parser.add_argument("--seed", type=int, default=_SAMPLING_SEED)
    parser.add_argument("--suffix", default=HiddenSuffixConfig().suffix, help="Hidden suffix including its boundary punctuation or space")
    parser.add_argument("--asr-model", help="Existing faster-whisper model directory, or a model name (default: small)")
    parser.add_argument("--alignment-model", help="Vietnamese character CTC model directory or model ID")
    parser.add_argument("--reuse-before", action="store_true", help="Re-align saved before_cut WAVs without running TTS again")
    parser.add_argument("--output", type=Path, default=ROOT / "models/zerotts/hidden-suffix-test")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--text", action="append", help="Test this sentence instead of the default list; may be repeated")
    args = parser.parse_args()
    if not 0 <= args.seed <= 4294967295:
        parser.error("seed must be from 0 to 4294967295")
    os.environ["ZEROTTS_ENABLE_HIDDEN_SUFFIX"] = "1"
    os.environ["ZEROTTS_HIDDEN_SUFFIX"] = args.suffix
    os.environ["ZEROTTS_SUFFIX_DEBUG"] = "1" if args.debug else "0"
    # Reuse diagnostic models already present in this workspace, if any.
    # Production model selection remains explicit in HiddenSuffixConfig.
    if not args.asr_model and not os.getenv("ZEROTTS_SUFFIX_ASR_MODEL"):
        cached = sorted((ROOT / "models/diagnostics/whisper/models--Systran--faster-whisper-small/snapshots").glob("*/model.bin"))
        if cached:
            args.asr_model = str(cached[-1].parent)
    if not args.alignment_model and not os.getenv("ZEROTTS_SUFFIX_ALIGN_MODEL"):
        cached_alignment = ROOT / "models/diagnostics/alignment/vietnamese-250h"
        if (cached_alignment / "config.json").is_file():
            args.alignment_model = str(cached_alignment)
    if args.asr_model:
        os.environ["ZEROTTS_SUFFIX_ASR_MODEL"] = args.asr_model
    if args.alignment_model:
        os.environ["ZEROTTS_SUFFIX_ALIGN_MODEL"] = args.alignment_model
    for name in ["before_cut", "after_cut", "legacy"]:
        (args.output / name).mkdir(parents=True, exist_ok=True)
    if args.reuse_before:
        config = HiddenSuffixConfig.from_env()
        detector = WhisperSuffixDetector(config.asr_model)
        saved = json.loads((args.output / "results.json").read_text(encoding="utf-8"))
        for row in saved:
            before, sample_rate = sf.read(args.output / "before_cut" / row["file"], dtype="float32")
            raw = (args.output / "before_cut" / row["file"]).with_suffix(".npy")
            if raw.is_file():
                before = np.load(raw, allow_pickle=False)
            for key in ["reason", "detection"]:
                row.pop(key, None)
            try:
                point = detector.detect(before, sample_rate, row["original_text"], config)
                audio, metadata = cut_at_boundary(before, sample_rate, point, config)
                row.update(metadata, status="trimmed", detection=detector.last_diagnostic)
            except Exception as exc:
                row.update(status="fallback", reason=str(exc), detection=detector.last_diagnostic)
                audio, _ = sf.read(args.output / "legacy" / (row["file"][:2] + ".wav"), dtype="float32")
            sf.write(args.output / "after_cut" / row["file"], audio, sample_rate, subtype="PCM_16")
            sf.write(args.output / "after_cut" / (row["file"][:2] + "_final.wav"), audio, sample_rate, subtype="PCM_16")
            print(f"{row['file']}: {row['status']} {row.get('cut_point', row.get('reason'))}", flush=True)
        (args.output / "results.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    backend = ZeroTTSBackend()
    results = []
    index = 0
    segment_index = 0

    def trace(report, before, after):
        nonlocal segment_index
        segment_index += 1
        name = f"{index:02d}_segment{segment_index}.wav"
        sf.write(args.output / "before_cut" / name, before.reshape(-1), report["sample_rate"], subtype="PCM_16")
        # Replay the exact samples ASR saw; PCM16 rounding can change a short
        # word's recognition even when the WAV sounds identical to a listener.
        np.save((args.output / "before_cut" / name).with_suffix(".npy"), np.asarray(before, dtype=np.float32).reshape(-1))
        sf.write(args.output / "after_cut" / name, after.reshape(-1), report["sample_rate"], subtype="PCM_16")
        report.update(file=name, voice=args.voice, seed=args.seed)
        results.append(report)
        (args.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    backend._suffix_trace_hook = trace
    sentences = args.text or SENTENCES
    for index, sentence in enumerate(sentences, 1):
        segment_index = 0
        # Bypass audio caches so every run exercises suffix detection.
        wav = backend._synthesize_uncached(sentence, args.voice, seed=args.seed)
        (args.output / "after_cut" / f"{index:02d}_final.wav").write_bytes(wav)
        os.environ["ZEROTTS_ENABLE_HIDDEN_SUFFIX"] = "0"
        try:
            original = backend._synthesize_uncached(sentence, args.voice, seed=args.seed)
            (args.output / "legacy" / f"{index:02d}.wav").write_bytes(original)
        finally:
            os.environ["ZEROTTS_ENABLE_HIDDEN_SUFFIX"] = "1"
        print(f"Completed {index}/{len(sentences)}: {results[-1]['status']}", flush=True)
    trimmed = sum(r["status"] == "trimmed" for r in results)
    print(f"Trimmed {trimmed}/{len(results)} segments. Outputs: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
