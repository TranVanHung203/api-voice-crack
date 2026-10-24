"""Preview the registered boy-like voices from the model's official WAV samples."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download
import soundfile as sf
from tts_delivery import change_wav_voice_color

MODEL = "zeroweight-ai/ZeroTTS"
REVISION = "c2bfbd67dc648cac455077333f7cf5c18a2e3bb4"


def main():
    destination = ROOT / "samples/zerotts-boys-proposals"
    destination.mkdir(parents=True, exist_ok=True)
    report = {"status": "registered_locally_not_deployed", "model": MODEL,
              "revision": REVISION, "profiles": []}
    cache = ROOT / "models/zerotts/cache" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
    for voice_id, pitch in (("quangminh", 5.0), ("tiendat", 4.0)):
        source_path = cache / "voices" / voice_id / "preview.wav"
        if not source_path.is_file():
            source_path = Path(hf_hub_download(MODEL, f"voices/{voice_id}/preview.wav", revision=REVISION))
        source = source_path.read_bytes()
        original = destination / f"{voice_id}-original.wav"
        original.write_bytes(source)
        audio = change_wav_voice_color(source, destination / "cache", pitch_semitones=pitch)
        target = destination / f"{voice_id}-child.wav"
        target.write_bytes(audio)
        report["profiles"].append({"proposed_id": voice_id + "_child", "source_voice_id": voice_id,
                                   "pitch_semitones": pitch, "tempo": 1.0,
                                   "duration": sf.info(target).duration, "sample": str(target)})
    (destination / "proposals.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
