"""Recover a real downloaded MP3 and serve it after a simulated rebuild."""

import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import numpy as np
import pytest
import soundfile as sf

from audio_cache_store import atomic_write
from tools import backup_zerotts_cache as backup
from tts_delivery import _encode_mp3_data, encode_mp3
import tts_delivery
from zerotts_backend import ZeroTTSBackend


def receipt(output, text="Xin chao"):
    wav = BytesIO()
    sf.write(wav, np.sin(np.arange(12000) * 2 * np.pi * 440 / 48000) * 0.1,
             48000, format="WAV", subtype="PCM_16")
    mp3 = _encode_mp3_data(wav.getvalue())
    item = {"id": "a" * 64, "status": "ok", "text": text, "bytes": len(mp3),
            "sha256": hashlib.sha256(mp3).hexdigest(),
            "payload": {"voice_id": backup.VOICE_ID, "text": text, "format": "mp3"}}
    atomic_write(output / "audio" / (item["id"] + ".mp3"), mp3)
    (output / "progress.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
    return item, mp3


def test_recovery_serves_original_mp3_without_synthesis_after_rebuild(monkeypatch, tmp_path):
    output = tmp_path / "warmup"
    item, mp3 = receipt(output)
    files = {}
    commits = []
    repo = "owner/cache"
    space = "owner/space"
    prefix = f"voice_packs/zerotts/{backup.VOICE_ID}/"
    space_files = {name: (backup.ROOT / name).read_bytes() for name in
                   ("zerotts_backend.py", "zerotts_suffix.py", "tts_delivery.py", "hf_app.py")}
    space_files[prefix + "voice.npz"] = b"voice fingerprint only; no model needed"

    class API:
        def get_space_variables(self, *args):
            return {"HF_AUDIO_CACHE_REPO": SimpleNamespace(value=repo),
                    "ZEROTTS_ENABLE_HIDDEN_SUFFIX": SimpleNamespace(value="1")}

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(private=True)

        def list_repo_files(self, repo_id, **kwargs):
            return list(space_files if repo_id == space else files)

        def create_commit(self, **kwargs):
            commits.append(kwargs)
            for operation in kwargs["operations"]:
                files[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()

        def get_paths_info(self, repo_id, paths, **kwargs):
            result = []
            for name in paths:
                if name in files:
                    data = files[name]
                    result.append(SimpleNamespace(path=name, size=len(data), lfs=None,
                                  blob_id=hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()))
            return result

    def download(repo_id, filename=None, **kwargs):
        filename = filename or kwargs["filename"]
        data = (space_files if repo_id == space else files)[filename]
        target = tmp_path / "hub" / repo_id / filename
        atomic_write(target, data)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert backup.main(["--space-id", space, "--output", str(output)]) == 0
    assert len(commits) == 2  # one batch for both audio formats, then verified manifest
    report = json.loads((output / "recovery/backup-report.json").read_text())
    assert report["complete"] and report["verified_audio_files"] == 2

    # Re-running verifies remote content and never overwrites an existing WAV.
    original_wav = next(data for name, data in files.items() if name.endswith(".wav"))
    assert backup.main(["--space-id", space, "--output", str(output)]) == 0
    assert len(commits) == 3  # only refresh the manifest
    assert next(data for name, data in files.items() if name.endswith(".wav")) == original_wav

    # A small runtime inventory must retain the original inventory's manifest.
    original_manifests = {name for name in files if name.startswith("recovery/")}
    variant_output = tmp_path / "runtime-variants"
    receipt(variant_output, text="Xin chao. Hello.")
    assert backup.main(["--space-id", space, "--output", str(variant_output)]) == 0
    updated_manifests = {name for name in files if name.startswith("recovery/")}
    assert original_manifests < updated_manifests
    assert len(updated_manifests) == len(original_manifests) + 1

    settings = backup.entry_defaults(space_files["hf_app.py"].decode())
    settings["ZEROTTS_ENABLE_HIDDEN_SUFFIX"] = "1"
    with backup.deployment_environment(settings):
        monkeypatch.setenv("HF_AUDIO_CACHE_REPO", repo)
        monkeypatch.setenv("ZEROTTS_AUDIO_CACHE_DIR", str(tmp_path / "fresh-container"))
        monkeypatch.setattr(ZeroTTSBackend, "_voice_roots", classmethod(
            lambda cls: (("bundled", output / "recovery/deployed-voices"),)))
        backend = ZeroTTSBackend()
        monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **kw: pytest.fail("Must use recovered WAV"))
        audio = backend.synthesize_wav(item["text"], backup.VOICE_ID)
        assert audio == original_wav
        monkeypatch.setattr(tts_delivery, "_encode_mp3_data", lambda *a: pytest.fail("Must use original MP3"))
        with tts_delivery._MP3_CACHE_LOCK:
            tts_delivery._MP3_CACHE.clear()
        assert encode_mp3(audio, tmp_path / "fresh-container") == mp3


def test_corrupt_download_is_not_backed_up(tmp_path):
    item, _ = receipt(tmp_path)
    (tmp_path / "audio" / (item["id"] + ".mp3")).write_bytes(b"ID3corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        backup.successful_records(tmp_path)


def test_environment_is_restored_even_on_failure(monkeypatch):
    monkeypatch.setenv("ZEROTTS_THREADS", "7")
    monkeypatch.delenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", raising=False)
    with pytest.raises(RuntimeError):
        with backup.deployment_environment({"ZEROTTS_THREADS": "2", "ZEROTTS_ENABLE_HIDDEN_SUFFIX": "1"}):
            assert backup.os.environ["ZEROTTS_THREADS"] == "2"
            raise RuntimeError()
    assert backup.os.environ["ZEROTTS_THREADS"] == "7"
    assert "ZEROTTS_ENABLE_HIDDEN_SUFFIX" not in backup.os.environ


@pytest.mark.parametrize("text", [" Năm ", "Trên tờ lịch này có", "Đây là gang tay", "Xin chao"])
def test_recovery_key_matches_backend_request_options(monkeypatch, text):
    monkeypatch.setenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", "1")
    backend = ZeroTTSBackend()
    observed = []

    def key(text, voice, options):
        value = json.dumps([text, voice, options], sort_keys=True)
        observed.append(value)
        return hashlib.sha256(value.encode()).hexdigest()

    monkeypatch.setattr(backend, "_audio_cache_key", key)
    expected = backup.cache_key(backend, text)
    observed.clear()
    monkeypatch.setattr(backend, "_read_cached_audio", lambda path: b"cached wav")
    monkeypatch.setattr(backend, "_deliver_audio", lambda text, voice, audio: audio)
    backend.synthesize_wav(text, backup.VOICE_ID)
    assert hashlib.sha256(observed[0].encode()).hexdigest() == expected
