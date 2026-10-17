"""Compare identical generated codes through full and streaming codec paths."""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zerotts_backend import ZeroTTSBackend, _MIN_FRAMES, _SAMPLING_SEED
from probe_zerotts_endings import SENTENCES


def main():
    folder = ROOT / "models/zerotts/codec-check"
    folder.mkdir(parents=True, exist_ok=True)
    engine = ZeroTTSBackend()._load_engine()
    voice = "VpYLlCQEgcoHgycJstNP"
    report = []
    for slug in ["kham_pha", "con_ca"]:
        text = SENTENCES[slug]
        stops = []
        original = engine._local_decode_frame

        def decode(*args, **kwargs):
            ctrl, codes = original(*args, **kwargs)
            stops.append(ctrl == engine.eoa_id)
            return ctrl, codes

        engine._local_decode_frame = decode
        try:
            np.random.seed(_SAMPLING_SEED)
            frames = list(engine._generate_frames(
                text, min_frames=_MIN_FRAMES, max_frames=1500,
                voice_emb=engine.resolve_voice(voice), eoa_extra_frames=8,
            ))
        finally:
            engine._local_decode_frame = original
        base_count = stops.index(True) + 1
        codes = np.stack(frames, axis=-1)
        base = engine.codec.decode(codes[:, :, :base_count]).reshape(-1)
        extended = engine.codec.decode(codes).reshape(-1)
        sf.write(folder / f"{slug}_full.wav", base, engine.sample_rate, subtype="PCM_16")
        sf.write(folder / f"{slug}_tail8.wav", extended, engine.sample_rate, subtype="PCM_16")
        row = {
            "text": text, "base_frames": base_count,
            "tail8_prefix_max_difference": float(np.max(np.abs(base - extended[:len(base)]))),
            "streaming": [],
        }
        for chunk in [1, 4, 16]:
            decoder = engine.codec.streaming_decoder()
            try:
                parts = [decoder.decode_chunk(codes[:, :, start:min(start + chunk, base_count)])
                         for start in range(0, base_count, chunk)]
            finally:
                decoder.close()
            stream = np.concatenate(parts, axis=-1).reshape(-1)
            sf.write(folder / f"{slug}_stream{chunk}.wav", stream, engine.sample_rate, subtype="PCM_16")
            row["streaming"].append({
                "chunk_frames": chunk, "samples": len(stream),
                "max_difference": float(np.max(np.abs(base - stream))) if len(base) == len(stream) else None,
            })
        report.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    (folder / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
