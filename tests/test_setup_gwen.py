"""Provisioning must isolate dependencies and recover after interrupted installs."""

import json
import os
from pathlib import Path
import subprocess

import pytest

import setup_gwen as setup


@pytest.fixture
def installer(monkeypatch, tmp_path):
    monkeypatch.delenv("GWEN_PYTHON", raising=False)
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    (tmp_path / "requirements-gwen.txt").write_text("qwen-tts==0.1.1\n", encoding="utf-8")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["-m", "venv"]:
            directory = Path(command[-1])
            executable = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.touch()

    monkeypatch.setattr(setup, "run", run)
    return tmp_path, calls, run


def test_isolated_install_is_validated_and_reused(installer):
    root, calls, _ = installer
    executable = setup.ensure_runtime()
    pip_calls = [call for call in calls if call[1:3] == ["-m", "pip"]]
    assert pip_calls and all(call[0] == str(executable) for call in pip_calls)
    assert "--system-site-packages" not in calls[0]
    assert "--index-url" in pip_calls[1] and "torch==2.8.0+cpu" in pip_calls[1]
    marker = root / ".venv-gwen/.gwen-runtime.json"
    assert json.loads(marker.read_text())["requirements_sha256"]
    calls.clear()
    assert setup.ensure_runtime() == executable
    assert calls == [[str(executable), "-c", setup.CHECK]]


def test_failed_install_is_not_ready_and_retries(installer, monkeypatch):
    root, calls, successful_run = installer

    def fail(command, **kwargs):
        successful_run(command, **kwargs)
        if "pip" in command:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(setup, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        setup.ensure_runtime()
    assert not (root / ".venv-gwen/.gwen-runtime.json").exists()
    monkeypatch.setattr(setup, "run", successful_run)
    setup.ensure_runtime()
    assert (root / ".venv-gwen/.gwen-runtime.json").exists()


def test_custom_runtime_is_validated_without_installing(installer, monkeypatch):
    root, calls, _ = installer
    custom = str(root / "custom/python")
    monkeypatch.setenv("GWEN_PYTHON", custom)
    assert setup.ensure_runtime() == Path(custom).resolve()
    assert calls == [[str(Path(custom).resolve()), "-c", setup.CHECK]]
    assert not (root / ".venv-gwen").exists()


def test_installer_removes_api_import_paths(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "api-specific-modules")
    monkeypatch.setenv("PYTHONHOME", "api-python-home")
    monkeypatch.setenv("VIRTUAL_ENV", "api-environment")
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    captured = {}
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs: captured.update(kwargs))
    setup.run(["isolated-python", "-c", "pass"])
    assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & captured["env"].keys()
    assert captured["env"]["PYTHONNOUSERSITE"] == "1"
    assert captured["env"]["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    assert captured["check"] is True


def test_space_downloads_only_after_validating_runtime(installer, monkeypatch):
    root, calls, _ = installer
    monkeypatch.delenv("GWEN_DEVICE", raising=False)
    setup.prepare_space_runtime()
    executable = Path(os.environ["GWEN_PYTHON"])
    assert executable.is_file()
    assert calls[-2] == [str(executable), "-c", setup.CHECK]
    assert "prepare_model" in calls[-1][-1]
    assert os.environ["GWEN_DEVICE"] == "cpu"
    # Avoid leaking settings into other tests.
    monkeypatch.delenv("GWEN_PYTHON")
