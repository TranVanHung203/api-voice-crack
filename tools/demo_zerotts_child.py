"""Preview childlike delivery from an existing downloaded voice, without TTS."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf
from huggingface_hub import hf_hub_download
from tts_delivery import change_wav_voice_color
from tools.backup_zerotts_cache import cache_key, deployment_environment
from zerotts_backend import ZeroTTSBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long", action="store_true", help="Preview a longer sentence")
    args = parser.parse_args()
    source_root = ROOT / "models/zerotts_cache_warmup/VpYLlCQEgcoHgycJstNP"
    records = [json.loads(line) for line in (source_root / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    text = ("Vẫn còn một chú thỏ chưa có cà rốt. Vì sao vậy nhỉ? Chúng ta hãy cùng khám phá nhé!"
            if args.long else "Hãy cùng khám phá nhé!")
    record = next(item for item in reversed(records) if item.get("status") == "ok" and item.get("text") == text)
    manifest = json.loads((source_root / "recovery/manifest.json").read_text(encoding="utf-8"))
    source_record = next(item for item in manifest["texts"] if item["text"] == text)
    local_wav = source_root / "recovery/wav" / Path(source_record["wav"]).name
    if local_wav.is_file():
        source = local_wav.read_bytes()
    else:
        report = json.loads((source_root / "recovery/backup-report.json").read_text(encoding="utf-8"))
        source = Path(hf_hub_download(report["dataset"], source_record["wav"], repo_type="dataset")).read_bytes()
    dest = ROOT / "samples" / ("zerotts-child-v2-long" if args.long else "zerotts-child-v2")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "original.wav").write_bytes(source)
    profiles = [("soft", 2.5, None), ("child", 3.0, None), ("higher", 4.0, None)]
    metadata = {"text": record["text"], "source_voice_id": record["payload"]["voice_id"],
                "source": "dataset-wav", "algorithm": "resample_atempo_v2", "profiles": []}
    for name, pitch, formant in profiles:
        audio = change_wav_voice_color(source, dest / "cache", pitch_semitones=pitch, formant_ratio=formant)
        path = dest / (name + ".wav")
        path.write_bytes(audio)
        metadata["profiles"].append({"name": name, "pitch_semitones": pitch,
                                     "formant_ratio": formant, "seconds": sf.info(path).duration})
    (dest / "comparison.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # Check all previously restored original-voice keys remain unchanged.

    class Deployed(ZeroTTSBackend):
        @classmethod
        def _voice_roots(cls):
            return (("bundled", source_root / "recovery/deployed-voices"),)

    with deployment_environment(manifest["settings"]):
        backend = Deployed()
        unchanged = sum(Path(item["wav"]).stem == cache_key(backend, item["text"]) for item in manifest["texts"])
    print(json.dumps({"samples": str(dest), "profiles": metadata["profiles"],
                      "original_cache_keys_unchanged": unchanged, "original_cached_texts": len(manifest["texts"])}))


if __name__ == "__main__":
    main()
