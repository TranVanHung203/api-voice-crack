"""Compare end-of-speech strategies without changing the API or installed package."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zerotts_backend import ZeroTTSBackend, _MIN_FRAMES, _SAMPLING_SEED


SENTENCES = {
    "kham_pha": "Hãy cùng khám phá nhé!",
    "con_ca": "Trong bể có bao nhiêu con cá?",
    "con_ca_vay": "Trong bể có bao nhiêu con cá vậy?",
    "kham_pha_nao": "Hãy cùng khám phá nhé nào!",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices", nargs="+", default=["hamy", "VpYLlCQEgcoHgycJstNP"])
    parser.add_argument("--variants", nargs="+", default=["baseline", "tail4", "debounce3", "control02"])
    parser.add_argument("--seed", type=int, default=_SAMPLING_SEED)
    parser.add_argument("--sentences", nargs="+", choices=list(SENTENCES), default=list(SENTENCES))
    parser.add_argument("--output", type=Path, default=ROOT / "models/zerotts/ending-comparison")
    args = parser.parse_args()
    supported = {"baseline", "tail4", "control02", "cfg13", "cfg15", "temp06", "repeat10", "repeat11"}
    for variant in args.variants:
        if variant in supported:
            continue
        if re.fullmatch(r"seed\d+", variant) and int(variant[4:]) <= 4294967295:
            continue
        if re.fullmatch(r"debounce\d+", variant) and 1 <= int(variant[8:]) < 12:
            continue
        parser.error(f"Unsupported variant: {variant}")
    args.output.mkdir(parents=True, exist_ok=True)
    backend = ZeroTTSBackend()
    engine = backend._load_engine()
    original_decode = engine._local_decode_frame
    results = []
    for voice in args.voices:
        for slug in args.sentences:
            text = SENTENCES[slug]
            for variant in args.variants:
                events = []
                state = {"frame": -1, "first": None}
                debounce = int(variant.removeprefix("debounce")) if variant.startswith("debounce") else 0

                def decode(h, forbid_eoa, **options):
                    state["frame"] += 1
                    frame, first = state["frame"], state["first"]
                    cooling = first is not None and frame <= first + debounce
                    ctrl, codes = original_decode(h, forbid_eoa=forbid_eoa or cooling, **options)
                    if ctrl == engine.eoa_id:
                        events.append({"frame": frame, "event": "eoa"})
                        if debounce and first is None:
                            state["first"] = frame
                            ctrl = engine.slot_id
                    if debounce and first is not None and frame >= first + 12 and ctrl != engine.eoa_id:
                        events.append({"frame": frame, "event": "wait_limit"})
                        ctrl = engine.eoa_id
                    return ctrl, codes

                engine._local_decode_frame = decode
                try:
                    seed = int(variant.removeprefix("seed")) if variant.startswith("seed") else args.seed
                    np.random.seed(seed)
                    timing = {}
                    start = time.perf_counter()
                    audio = engine.synthesize(
                        text, voice=voice, min_frames=_MIN_FRAMES, max_frames=1500,
                        eoa_extra_frames=4 if variant == "tail4" else 1,
                        text_temperature=0.2 if variant == "control02" else 1.0,
                        cfg_scale={"cfg13": 1.3, "cfg15": 1.5}.get(variant, 1.0),
                        audio_temperature=0.6 if variant == "temp06" else 0.8,
                        audio_repetition_penalty={"repeat10": 1.0, "repeat11": 1.1}.get(variant, 1.2),
                        timing=timing,
                    ).reshape(-1)
                    seconds = time.perf_counter() - start
                finally:
                    engine._local_decode_frame = original_decode
                fixed = backend._shorten_tail_pause(audio, engine.sample_rate, 80)
                name = f"{voice}_{slug}_{variant}.wav"
                sf.write(args.output / name, audio, engine.sample_rate, subtype="PCM_16")
                row = {
                    "file": name, "text": text, "voice": voice, "variant": variant,
                    "seed": seed, "seconds": round(seconds, 3),
                    "duration": len(audio) / engine.sample_rate, "frames": timing["n_frames"],
                    "postprocess_removed_samples": len(audio) - np.asarray(fixed).size,
                    "events": events,
                }
                results.append(row)
                (args.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(row, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
