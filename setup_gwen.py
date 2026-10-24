"""Install and validate the isolated CPU Gwen runtime (Linux/Spaces or Windows)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CHECK = (
    "import torch, torchaudio, qwen_tts, sea_g2p; "
    "from importlib.metadata import version; "
    "assert version('transformers') == '4.57.3'; "
    "print('Gwen runtime OK; torch=' + torch.__version__, flush=True)"
)


def run(command: list[str], *, timeout: int = 1200) -> None:
    # Do not inherit the API's import paths into the isolated installer/worker.
    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               HF_HUB_ENABLE_HF_TRANSFER="0")
    subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=timeout)


def ensure_runtime(directory: Path | None = None) -> Path:
    """Retry incomplete installs; mark ready only after actual SDK imports pass."""
    if os.getenv("GWEN_PYTHON"):
        executable = Path(os.environ["GWEN_PYTHON"]).resolve()
        run([str(executable), "-c", CHECK], timeout=180)
        return executable

    directory = (directory or ROOT / ".venv-gwen").resolve()
    executable = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirements = ROOT / "requirements-gwen.txt"
    expected = {
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
        "python": list(sys.version_info[:2]),
        "torch_index": "https://download.pytorch.org/whl/cpu",
    }
    marker = directory / ".gwen-runtime.json"
    try:
        ready = executable.is_file() and json.loads(marker.read_text(encoding="utf-8")) == expected
    except (OSError, ValueError):
        ready = False
    if ready:
        try:
            run([str(executable), "-c", CHECK], timeout=180)
            return executable
        except (OSError, subprocess.SubprocessError):
            pass  # Repair a damaged environment rather than trusting its marker.

    marker.unlink(missing_ok=True)
    if not executable.is_file():
        print("Gwen: creating isolated Python environment", flush=True)
        run([sys.executable, "-m", "venv", str(directory)])
    print("Gwen: installing CPU runtime; this can take several minutes", flush=True)
    run([str(executable), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(executable), "-m", "pip", "install", "torch==2.8.0+cpu", "torchaudio==2.8.0+cpu",
         "--index-url", expected["torch_index"]])
    run([str(executable), "-m", "pip", "install", "-r", str(requirements)])
    run([str(executable), "-c", CHECK], timeout=180)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(expected), encoding="utf-8")
    temporary.replace(marker)
    return executable


def prepare_space_runtime() -> None:
    """Complete installation/downloads before the Space accepts TTS requests."""
    os.environ.setdefault("GWEN_DEVICE", "cpu")
    os.environ.setdefault("GWEN_THREADS", "2")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    executable = ensure_runtime()
    os.environ["GWEN_PYTHON"] = str(executable)
    print("Gwen: preparing pinned model weights", flush=True)
    run([str(executable), "-c",
         "import os; from gwen_worker import prepare_model; "
         "from gwen_backend import MODEL_ID, MODEL_REVISION; "
         "model = os.getenv('GWEN_MODEL_ID', MODEL_ID); "
         "revision = os.getenv('GWEN_MODEL_REVISION') or (MODEL_REVISION if model == MODEL_ID else None); "
         "print(prepare_model(model, revision), flush=True)"], timeout=1200)
    print("Gwen: isolated runtime and model files ready", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-model", action="store_true")
    args = parser.parse_args()
    if args.download_model:
        prepare_space_runtime()
    else:
        print(ensure_runtime())
