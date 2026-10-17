"""Optional offline ASR check of probe WAVs; recognition is not a listening test."""

import argparse
import fnmatch
import json
from pathlib import Path

from faster_whisper import WhisperModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pattern", default="*.wav")
    parser.add_argument("--report", default="asr.json")
    args = parser.parse_args()
    model = WhisperModel(str(args.model), device="cpu", compute_type="int8", cpu_threads=4)
    results = []
    for row in json.loads((args.folder / "results.json").read_text(encoding="utf-8")):
        if not fnmatch.fnmatch(row["file"], args.pattern):
            continue
        segments, _ = model.transcribe(
            str(args.folder / row["file"]), language="vi", beam_size=5,
            condition_on_previous_text=False, vad_filter=False, word_timestamps=True,
        )
        segments = list(segments)
        row["asr"] = "".join(s.text for s in segments).strip()
        row["words"] = [
            {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
            for s in segments for w in (s.words or [])
        ]
        results.append(row)
        (args.folder / args.report).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"file": row["file"], "asr": row["asr"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
