"""JSON-line worker; run only with the isolated Gwen Python environment."""

from __future__ import annotations

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import re
import sys
import traceback
import unicodedata

from gwen_backend import MODEL_ID, MODEL_REVISION, ROOT, resolve_voice

# Recommended by ggroup-ai-lab/gwen-tts/inference.py at f0f24d8.
GENERATION_CONFIG = dict(
    temperature=0.3, top_k=20, top_p=0.9, max_new_tokens=4096,
    repetition_penalty=2.0, subtalker_do_sample=True,
    subtalker_temperature=0.1, subtalker_top_k=20, subtalker_top_p=1.0,
)
MODEL_FILES = ["config.json", "generation_config.json", "merges.txt", "model.safetensors",
               "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "speech_tokenizer/*"]


def prepare_model(model_id: str, revision: str | None) -> str:
    if Path(model_id).is_dir() or model_id != MODEL_ID:
        return model_id
    # A flat snapshot works on Windows without symlink privileges. It also
    # gives the SDK one complete directory for the backbone and speech codec.
    from huggingface_hub import snapshot_download

    directory = ROOT / "models/gwen/model"
    marker = directory / ".gwen-revision.json"
    expected = {"model_id": model_id, "revision": revision}
    try:
        ready = json.loads(marker.read_text(encoding="utf-8")) == expected
    except (OSError, ValueError):
        ready = False
    if not ready:
        snapshot_download(model_id, revision=revision, local_dir=directory,
                          allow_patterns=MODEL_FILES, max_workers=1)
        marker.write_text(json.dumps(expected), encoding="utf-8")
    return str(directory)


def text_chunks(text: str, max_chars: int = 320) -> list[str]:
    """Normalize whitespace and split at punctuation/words without dropping text."""
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
    chunks, current = [], ""
    for sentence in re.split(r"(?<=[.!?;。！？])\s+", text):
        for word in sentence.split():
            if current and len(current) + len(word) + 1 > max_chars:
                chunks.append(current)
                current = ""
            while len(word) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(word[:max_chars])
                word = word[max_chars:]
            current = f"{current} {word}".strip()
        if current:
            chunks.append(current)
            current = ""
    return chunks


class GwenEngine:
    def __init__(self):
        import torch
        from qwen_tts import Qwen3TTSModel
        from sea_g2p import Normalizer

        threads = int(os.getenv("GWEN_THREADS", "4"))
        if threads < 1:
            raise ValueError("GWEN_THREADS must be positive")
        torch.set_num_threads(threads)
        device = os.getenv("GWEN_DEVICE", "auto")
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype_name = os.getenv("GWEN_DTYPE", "auto")
        if dtype_name == "auto":
            dtype_name = "bfloat16" if device.startswith("cuda") else "float32"
        if dtype_name not in {"float32", "float16", "bfloat16"}:
            raise ValueError("GWEN_DTYPE must be auto, float32, float16, or bfloat16")
        model_id = os.getenv("GWEN_MODEL_ID", MODEL_ID)
        revision = os.getenv("GWEN_MODEL_REVISION") or (MODEL_REVISION if model_id == MODEL_ID else None)
        print(f"Loading Gwen {model_id}@{revision}, device={device}, dtype={dtype_name}", file=sys.stderr, flush=True)
        model_path = prepare_model(model_id, revision)
        self.model = Qwen3TTSModel.from_pretrained(
            model_path, revision=revision, device_map=device,
            dtype=getattr(torch, dtype_name), attn_implementation="sdpa",
        )
        self.normalizer = Normalizer(lang="vi")
        self.prompts = {}
        print("Gwen model ready", file=sys.stderr, flush=True)

    def synthesize(self, text: str, voice_id: str, output_path: Path):
        import numpy as np
        import soundfile as sf

        key, preset, reference = resolve_voice(voice_id)
        if key not in self.prompts:
            self.prompts[key] = self.model.create_voice_clone_prompt(
                ref_audio=str(reference), ref_text=preset["text"],
            )
        normalized = self.normalizer.normalize(text)
        chunks = text_chunks(normalized)
        if not chunks:
            raise ValueError("text contains no speakable content")
        parts, sample_rate = [], None
        for index, chunk in enumerate(chunks):
            print(f"Gwen chunk {index + 1}/{len(chunks)} ({len(chunk)} chars)", file=sys.stderr, flush=True)
            wavs, sr = self.model.generate_voice_clone(
                text=chunk, voice_clone_prompt=self.prompts[key], **GENERATION_CONFIG,
            )
            samples = np.asarray(wavs[0], dtype=np.float32)
            if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
                raise ValueError("model returned empty or invalid audio")
            if sample_rate is not None and sr != sample_rate:
                raise ValueError("model changed sample rate between chunks")
            if parts:
                parts.append(np.zeros(round(sr * 0.15), dtype=np.float32))
            parts.append(samples)
            sample_rate = sr
        sf.write(output_path, np.concatenate(parts), sample_rate, format="WAV", subtype="PCM_16")


def main():
    engine = None
    # Preserve stdout exclusively for the protocol, including during SDK imports.
    protocol = sys.stdout
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request["id"]
            with redirect_stdout(sys.stderr):
                if engine is None:
                    engine = GwenEngine()
                engine.synthesize(request["text"], request["voice_id"], Path(request["output_path"]))
            result = {"id": request_id, "ok": True}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            result = {"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        protocol.write(json.dumps(result, ensure_ascii=False) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
