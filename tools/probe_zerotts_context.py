"""Log real backend segments and EOA frames for a controlled continuation test.

Uses uncached synthesis and a fixed NumPy seed; changes no installed package,
model, API configuration, or generation defaults. Frame indices start at zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import zerotts_backend as settings
from zerotts_backend import ZeroTTSBackend


QUESTION = "Trong bể có bao nhiêu con cá?"
CONTINUATION = "Bạn biết không"
CASES = {
    "A_question": QUESTION,
    "B_no_space": QUESTION + CONTINUATION,
    "B_space": QUESTION + " " + CONTINUATION,
    "C_newline": QUESTION + "\n" + CONTINUATION,
    "direct_A": QUESTION,
    "direct_B": QUESTION + " " + CONTINUATION,
}


def rng_fingerprint():
    state = np.random.get_state()
    return hashlib.sha256(state[1].tobytes() + repr(state[2:]).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="VpYLlCQEgcoHgycJstNP")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=ROOT / "models/zerotts/context-probe")
    args = parser.parse_args()
    if not 0 <= args.seed <= 4294967295:
        parser.error("seed must be from 0 to 4294967295")
    args.output.mkdir(parents=True, exist_ok=True)
    backend = ZeroTTSBackend()
    engine = backend._load_engine()
    original_synthesize = engine.synthesize
    original_decode = engine._local_decode_frame
    results = {}
    raw_segments = {}
    logs = []
    current = None
    case_name = ""

    def emit(line):
        logs.append(line)
        print(line, flush=True)

    def decode(*call_args, **kwargs):
        ctrl, codes = original_decode(*call_args, **kwargs)
        if current is not None:
            frame = current["decode_calls"]
            current["decode_calls"] += 1
            if ctrl == engine.eoa_id:
                current["eoa_frames"].append(frame)
                emit(f"EOA case={case_name} segment={current['index']} frame={frame}")
        return ctrl, codes

    def synthesize(segment, *call_args, **kwargs):
        nonlocal current
        current = {
            "index": len(results[case_name]["segments"]) + 1,
            "text": segment, "request_seed": args.seed,
            "rng_before": rng_fingerprint(), "decode_calls": 0, "eoa_frames": [],
        }
        emit(f"TTS SEGMENT case={case_name} seed={args.seed}: {segment!r}")
        timing = {}
        audio = original_synthesize(segment, *call_args, timing=timing, **kwargs)
        current["generated_frames"] = timing["n_frames"]
        current["samples"] = int(audio.size)
        current["raw_sha256"] = hashlib.sha256(audio.tobytes()).hexdigest()
        raw_segments.setdefault(case_name, []).append(audio.copy())
        raw_file = f"{case_name}_segment{current['index']}_raw.wav"
        sf.write(args.output / raw_file, audio.reshape(-1), engine.sample_rate, subtype="PCM_16")
        current["file"] = raw_file
        results[case_name]["segments"].append(current)
        current = None
        return audio

    engine.synthesize = synthesize
    engine._local_decode_frame = decode
    try:
        for case_name, text in CASES.items():
            results[case_name] = {"input": text, "segments": [], "file": case_name + ".wav"}
            if case_name.startswith("direct_"):
                state = np.random.get_state()
                try:
                    np.random.seed(args.seed)
                    audio = engine.synthesize(
                        text, voice=args.voice, min_frames=settings._MIN_FRAMES,
                        max_frames=1500, eoa_extra_frames=1,
                        text_temperature=settings._TEXT_TEMPERATURE,
                    )
                    wav = backend._encode_wav(audio, engine.sample_rate)
                finally:
                    np.random.set_state(state)
            else:
                wav = backend._synthesize_uncached(text, args.voice, seed=args.seed)
            (args.output / results[case_name]["file"]).write_bytes(wav)
            (args.output / "segments.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
            (args.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        engine.synthesize = original_synthesize
        engine._local_decode_frame = original_decode

    baseline = raw_segments["A_question"][0]
    comparisons = {
        name: {
            "first_segment_identical_to_A": bool(np.array_equal(baseline, segments[0])),
            "same_initial_rng": results[name]["segments"][0]["rng_before"] == results["A_question"]["segments"][0]["rng_before"],
        }
        for name, segments in raw_segments.items() if name != "A_question"
    }
    for name in ["B_space", "C_newline"]:
        a, _ = sf.read(args.output / "A_question.wav", dtype="int16")
        b, _ = sf.read(args.output / f"{name}.wav", dtype="int16")
        comparisons[name]["final_wav_prefix_identical_to_A"] = bool(np.array_equal(a, b[:len(a)]))
    (args.output / "comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    emit(json.dumps(comparisons, ensure_ascii=True))
    (args.output / "segments.log").write_text("\n".join(logs) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
