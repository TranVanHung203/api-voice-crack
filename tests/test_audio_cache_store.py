"""Durable cache recovery after rebuilding a Space, without a live Hub."""

import asyncio
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import wave

import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError
import pytest

import audio_cache_store as store
from gwen_backend import GwenBackend
import server
import tts_delivery
from zerotts_backend import ZeroTTSBackend
from vtts_backend import VTTSBackend


def wav_bytes():
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * 100)
    return output.getvalue()


@pytest.fixture
def hub(monkeypatch, tmp_path):
    files = {}
    monkeypatch.setenv("HF_AUDIO_CACHE_REPO", "owner/private-cache")
    monkeypatch.setenv("HF_AUDIO_CACHE_TOKEN", "test-secret")

    class API:
        def __init__(self, token):
            assert token == "test-secret"

        def upload_file(self, **kwargs):
            assert kwargs["repo_id"] == "owner/private-cache"
            assert kwargs["repo_type"] == "dataset"
            files[kwargs["path_in_repo"]] = kwargs["path_or_fileobj"]

    def download(**kwargs):
        assert kwargs["token"] == "test-secret"
        assert kwargs["repo_type"] == "dataset"
        name = kwargs["filename"]
        if name not in files:
            raise EntryNotFoundError("Missing test entry")
        target = tmp_path / "hub" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[name])
        return str(target)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return files


def test_gwen_rebuild_recovers_without_starting_worker(hub, monkeypatch, tmp_path):
    monkeypatch.setenv("GWEN_AUDIO_CACHE_DIR", str(tmp_path / "first-container"))
    backend = GwenBackend()
    monkeypatch.setattr(backend, "_run_worker", lambda text, voice, path: path.write_bytes(wav_bytes()))
    audio = backend.synthesize_wav("Xin chao", "yen_nhi")
    assert len(hub) == 1
    # Simulate a completely fresh filesystem and Python process.
    monkeypatch.setenv("GWEN_AUDIO_CACHE_DIR", str(tmp_path / "rebuilt-container"))
    restarted = GwenBackend()
    monkeypatch.setattr(restarted, "_run_worker", lambda *a: pytest.fail("Must recover the remote cache"))
    assert restarted.synthesize_wav("Xin chao", "Yen-Nhi") == audio
    assert len(list((tmp_path / "rebuilt-container").glob("*.wav"))) == 1
    # The restored file remains usable while the Hub is unavailable.
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **k: pytest.fail("Local cache must win"))
    assert GwenBackend().synthesize_wav("Xin chao", "yen_nhi") == audio


def test_edge_rebuild_recovers_without_microsoft(hub, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_CACHE_DIR", tmp_path / "first")
    monkeypatch.setattr(server, "_audio_cache", OrderedDict())
    monkeypatch.setattr(server, "_cache_locks", {})

    class Communicate:
        def __init__(self, **kwargs):
            pass

        async def stream(self):
            yield {"type": "audio", "data": b"ID3edge-audio"}

    monkeypatch.setattr(server.edge_tts, "Communicate", Communicate)
    args = ("Xin chao", "vi-VN-HoaiMyNeural", "+0%", "+0%", "+0Hz")
    audio = asyncio.run(server._get_audio(*args))
    monkeypatch.setattr(server, "_CACHE_DIR", tmp_path / "rebuilt")
    monkeypatch.setattr(server, "_audio_cache", OrderedDict())
    monkeypatch.setattr(server.edge_tts, "Communicate", lambda **k: pytest.fail("Remote hit must skip Microsoft"))
    assert asyncio.run(server._get_audio(*args)) == audio


def test_mp3_rebuild_skips_encoding(hub, monkeypatch, tmp_path):
    monkeypatch.setattr(tts_delivery, "_MP3_CACHE", OrderedDict())
    monkeypatch.setattr(tts_delivery, "_encode_mp3_data", lambda wav: b"ID3encoded-audio")
    audio = tts_delivery.encode_mp3(wav_bytes(), tmp_path / "first")
    monkeypatch.setattr(tts_delivery, "_MP3_CACHE", OrderedDict())
    monkeypatch.setattr(tts_delivery, "_encode_mp3_data", lambda wav: pytest.fail("Must skip encoding"))
    assert tts_delivery.encode_mp3(wav_bytes(), tmp_path / "rebuilt") == audio


@pytest.mark.parametrize("registered", [False, True])
def test_vtts_rebuild_recovers_and_invalidates_changed_inputs(hub, monkeypatch, tmp_path, registered):
    monkeypatch.setenv("VTTS_MODEL_DIR", str(tmp_path / "vtts"))
    monkeypatch.setenv("VTTS_AUDIO_CACHE_DIR", str(tmp_path / "first"))
    asset = tmp_path / "vtts/voices/voice.npz" if registered else tmp_path / "reference.wav"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"reference data")
    backend = VTTSBackend()
    method = "synthesize_registered_voice" if registered else "synthesize_wav"
    uncached = "_synthesize_registered_voice_uncached" if registered else "_synthesize_wav_uncached"
    voice = "voice" if registered else asset
    monkeypatch.setattr(backend, uncached, lambda *a, **k: wav_bytes())
    audio = getattr(backend, method)("Hello", voice)
    monkeypatch.setenv("VTTS_AUDIO_CACHE_DIR", str(tmp_path / "rebuilt"))
    restarted = VTTSBackend()
    calls = []
    monkeypatch.setattr(restarted, uncached, lambda *a, **k: calls.append((a, k)) or wav_bytes())
    assert getattr(restarted, method)("Hello", voice) == audio
    assert not calls
    getattr(restarted, method)("Changed text", voice)
    getattr(restarted, method)("Hello", voice, length_scale=0.9)
    asset.write_bytes(b"changed reference")
    getattr(restarted, method)("Hello", voice)
    assert len(calls) == 3


def test_zerotts_cache_key_survives_relocation_and_timestamp_changes(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROTTS_REVISION", "a" * 40)
    monkeypatch.delenv("ZEROTTS_MODEL_DIR", raising=False)
    monkeypatch.setattr(ZeroTTSBackend, "_bundled_voices_dir", staticmethod(lambda: tmp_path / "bundled"))
    keys = []
    for name in ("first", "rebuilt"):
        root = tmp_path / name
        monkeypatch.setattr(ZeroTTSBackend, "_data_dir", staticmethod(lambda: root))
        voice = root / "voices/voice/voice.npz"
        voice.parent.mkdir(parents=True)
        voice.write_bytes(b"same voice data")
        # Hub weights may be present in only one of the two containers.
        if name == "first":
            model = root / "cache/models--test/snapshots/commit/model.onnx"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"pinned model")
        keys.append(ZeroTTSBackend()._audio_cache_key("Hello", "voice", {}))
    assert keys[0] == keys[1]
    voice.write_bytes(b"changed voice data")
    assert ZeroTTSBackend()._audio_cache_key("Hello", "voice", {}) != keys[0]


def test_zerotts_rebuild_hits_remote_before_model_load(hub, monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROTTS_REVISION", "a" * 40)
    monkeypatch.delenv("ZEROTTS_MODEL_DIR", raising=False)
    monkeypatch.setattr(ZeroTTSBackend, "_bundled_voices_dir", staticmethod(lambda: tmp_path / "bundled"))
    monkeypatch.setenv("ZEROTTS_DATA_DIR", str(tmp_path / "first"))
    backend = ZeroTTSBackend()
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **k: wav_bytes())
    audio = backend.synthesize_wav("Hello", "voice")
    monkeypatch.setenv("ZEROTTS_DATA_DIR", str(tmp_path / "rebuilt"))
    restarted = ZeroTTSBackend()
    monkeypatch.setattr(restarted, "_load_engine", lambda: pytest.fail("Must not load model"))
    assert restarted.synthesize_wav("Hello", "voice") == audio


def test_zerotts_fallback_is_recovered_from_dataset_after_rebuild(hub, monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROTTS_ENABLE_HIDDEN_SUFFIX", "1")
    monkeypatch.setenv("ZEROTTS_REVISION", "a" * 40)
    monkeypatch.delenv("ZEROTTS_MODEL_DIR", raising=False)
    monkeypatch.setattr(ZeroTTSBackend, "_bundled_voices_dir", staticmethod(lambda: tmp_path / "bundled"))
    monkeypatch.setenv("ZEROTTS_DATA_DIR", str(tmp_path / "first"))
    backend = ZeroTTSBackend()

    def fallback(*args, **kwargs):
        backend._suffix_fallback_used = True
        return wav_bytes()

    monkeypatch.setattr(backend, "_synthesize_uncached", fallback)
    first = backend.synthesize_wav("Hello", "voice")
    assert backend._suffix_fallback_used
    assert list(hub.values()) == [first]
    monkeypatch.setenv("ZEROTTS_DATA_DIR", str(tmp_path / "rebuilt"))
    restarted = ZeroTTSBackend()
    monkeypatch.setattr(restarted, "_synthesize_uncached", lambda *a, **k: pytest.fail("Remote fallback cache must skip synthesis and ASR"))
    assert restarted.synthesize_wav("Hello", "voice") == first


def test_cache_disabled_uses_only_local_files(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_AUDIO_CACHE_REPO", raising=False)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **k: pytest.fail("Cache is disabled"))
    path = tmp_path / "clip.wav"
    store.atomic_write(path, wav_bytes())
    assert store.read_audio(path, "gwen") == wav_bytes()
    assert not store.backup_audio(path, "gwen", wav_bytes())
    with pytest.raises(FileNotFoundError):
        store.read_audio(tmp_path / "missing.wav", "gwen")


def test_backup_retries_and_keeps_local_audio(hub, monkeypatch, tmp_path, caplog):
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("test-secret must never be logged")

    monkeypatch.setattr(huggingface_hub, "HfApi", fail)
    monkeypatch.setattr(store.time, "sleep", lambda seconds: None)
    path = tmp_path / "clip.wav"
    path.write_bytes(wav_bytes())
    assert not store.backup_audio(path, "gwen", wav_bytes())
    assert len(calls) == 3
    assert store.read_audio(path, "gwen") == wav_bytes()
    assert "only the local copy" in caplog.text
    assert "test-secret" not in caplog.text


def test_remote_outage_becomes_cache_miss(hub, monkeypatch, tmp_path, caplog):
    def fail(**kwargs):
        raise RuntimeError("test-secret")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    with pytest.raises(FileNotFoundError):
        store.read_audio(tmp_path / "missing.wav", "gwen")
    assert "generating audio" in caplog.text
    assert "test-secret" not in caplog.text


@pytest.mark.parametrize("status, hint", [
    (401, "authentication"),
    (403, "write permission"),
    (404, "repository and token access"),
    (409, "commit conflict"),
    (429, "rate limit"),
    (503, "server error"),
])
def test_backup_logs_http_status_without_credentials(hub, monkeypatch, tmp_path, caplog, status, hint):
    from requests import Response
    from huggingface_hub.errors import HfHubHTTPError

    response = Response()
    response.status_code = status
    response.headers["Retry-After"] = "60"
    response.headers["Authorization"] = "test-secret"
    response._content = b"test-secret response body"
    error = HfHubHTTPError("test-secret error message", response=response)

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(huggingface_hub, "HfApi", fail)
    monkeypatch.setattr(store.time, "sleep", lambda seconds: None)
    assert not store.backup_audio(tmp_path / "clip.wav", "zerotts", wav_bytes())
    assert f"HTTP {status}" in caplog.text
    assert hint in caplog.text
    assert "retry after 60 seconds" in caplog.text
    assert "test-secret" not in caplog.text


def test_deploy_backs_up_only_audio_before_configuring_space(monkeypatch, tmp_path):
    from tools.deploy_huggingface import configure_audio_cache

    monkeypatch.setattr(store, "ROOT", tmp_path)
    root = tmp_path / "models/gwen/audio_cache"
    (root / "encoded").mkdir(parents=True)
    (root / "aa.wav").write_bytes(wav_bytes())
    (root / "encoded/bb.mp3").write_bytes(b"ID3mp3")
    (root / "voice_color").mkdir()
    (root / "voice_color/cc.wav").write_bytes(wav_bytes())
    (root / "partial.tmp").write_bytes(b"incomplete")
    (root / "worker.log").write_text("private log")
    commits, secrets, variables = [], [], []
    api = SimpleNamespace(
        token="test-token",
        create_repo=lambda *a, **k: None,
        repo_info=lambda *a, **k: SimpleNamespace(private=True),
        list_repo_files=lambda *a, **k: [],
        create_commit=lambda **k: commits.append(k),
        add_space_secret=lambda **k: secrets.append(k),
        add_space_variable=lambda **k: variables.append(k),
    )
    assert configure_audio_cache(api, "owner/space", "owner/cache") == 3
    assert {operation.path_in_repo for operation in commits[0]["operations"]} == {
        "audio/gwen/aa/aa.wav", "audio/mp3/bb/bb.mp3", "audio/voice_color/cc/cc.wav",
    }
    assert secrets[0]["key"] == "HF_AUDIO_CACHE_TOKEN"
    assert secrets[0]["value"] == "test-token"
    assert variables[0]["value"] == "owner/cache"
    api.repo_info = lambda *a, **k: SimpleNamespace(private=False)
    with pytest.raises(SystemExit, match="must be private"):
        configure_audio_cache(api, "owner/space", "owner/cache")
    assert len(commits) == 1
