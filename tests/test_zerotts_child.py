"""A derived childlike voice reuses the original voice and durable caches."""

from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil

from fastapi.testclient import TestClient
import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError
import numpy as np
import pytest
import soundfile as sf

import server
import tts_delivery
import zerotts_backend as module
from zerotts_backend import ZeroTTSBackend, ZeroTTSBackendError

SOURCE = module._TUNED_VOICE_ID
CHILD = module._CHILD_VOICE_ID


def tone(frequency=220):
    output = BytesIO()
    rate = 48000
    sf.write(output, .2 * np.sin(2 * np.pi * frequency * np.arange(rate) / rate),
             rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


@pytest.fixture
def backend(monkeypatch, tmp_path):
    # Each case simulates its own machine/cache repository. Equal test tones
    # from different voices must not reuse a previous case's process MP3 cache.
    monkeypatch.setattr(tts_delivery, "_MP3_CACHE", type(tts_delivery._MP3_CACHE)())
    bundled = tmp_path / "bundled"
    for voice in (SOURCE, CHILD, *module._BOY_VOICES):
        pack = bundled / voice
        pack.mkdir(parents=True)
        (pack / "voice.npz").write_bytes(b"same source speaker embedding")
    monkeypatch.setattr(ZeroTTSBackend, "_bundled_voices_dir", staticmethod(lambda: bundled))
    monkeypatch.setenv("ZEROTTS_DATA_DIR", str(tmp_path / "first"))
    monkeypatch.setenv("ZEROTTS_REVISION", "b" * 40)
    monkeypatch.delenv("ZEROTTS_MODEL_DIR", raising=False)
    monkeypatch.delenv("ZEROTTS_AUDIO_CACHE_DIR", raising=False)
    monkeypatch.delenv("HF_AUDIO_CACHE_REPO", raising=False)
    return ZeroTTSBackend()


def test_bundled_copy_is_listed_and_retains_source_embedding(monkeypatch):
    monkeypatch.delenv("ZEROTTS_BUNDLED_VOICES_DIR", raising=False)
    root = module._DEFAULT_BUNDLED_VOICES_DIR
    for name in ("voice.npz", "voice.bin"):
        assert (root / SOURCE / name).read_bytes() == (root / CHILD / name).read_bytes()
    child = next(voice for voice in ZeroTTSBackend().list_voices() if voice["voice_id"] == CHILD)
    assert "childlike" in child["tags"]


def test_bundled_child_metadata_survives_windows_sdk_default_encoding():
    path = module._DEFAULT_BUNDLED_VOICES_DIR / CHILD / "meta.json"
    # ZeroTTS 0.1.5 calls read_text() without encoding. The child metadata
    # previously failed at byte 0x8d under the Windows cp1252 default.
    metadata = json.loads(path.read_bytes().decode("cp1252"))
    assert metadata == json.loads(path.read_text(encoding="utf-8"))
    assert metadata["display_name"] == "Chị Thắm – giọng trẻ hóa"


def test_child_reuses_original_cache_and_calibrated_seed(backend, monkeypatch):
    calls = []
    source, child = tone(), tone(280)
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda text, voice, **kw: calls.append((voice, kw)) or source)
    effects = []
    monkeypatch.setattr(tts_delivery, "change_wav_voice_color", lambda wav, *args, **kw: effects.append((wav, kw)) or child)
    assert backend.synthesize_wav("Năm", SOURCE) == source
    assert backend.synthesize_wav("Năm", CHILD) == child
    assert backend.synthesize_wav("Năm", SOURCE) == source
    assert len(calls) == 1 and calls[0][0] == SOURCE
    assert calls[0][1]["seed"] == 11
    assert effects == [(source, {"pitch_semitones": 3.0, "formant_ratio": None})]
    backend.synthesize_wav("Năm", CHILD, seed=7)
    assert calls[-1][1]["seed"] == 7


@pytest.mark.parametrize("voice_id,source_id,pitch", [(CHILD, SOURCE, 3.0),
    ("quangminh_child", "quangminh", 5.0), ("tiendat_child", "tiendat", 4.0),
    ("quangvinh_child", "htSzITlVXGL3s4XwNqOv", 4.0)])
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_real_child_pitch_duration_and_persistent_recovery(backend, monkeypatch, tmp_path, voice_id, source_id, pitch):
    files = {}
    monkeypatch.setenv("HF_AUDIO_CACHE_REPO", "owner/cache")

    class API:
        def __init__(self, **kwargs):
            pass

        def upload_file(self, **kwargs):
            files[kwargs["path_in_repo"]] = kwargs["path_or_fileobj"]

    def download(**kwargs):
        name = kwargs["filename"]
        if name not in files:
            raise EntryNotFoundError("Missing")
        path = tmp_path / "hub" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[name])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    generated = []
    source = tone()
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **kw: generated.append(a) or source)
    child = backend.synthesize_wav("Xin chao", voice_id)
    audio, rate = sf.read(BytesIO(child))
    spectrum = abs(np.fft.rfft(audio * np.hanning(len(audio))))
    frequency = np.fft.rfftfreq(len(audio), 1 / rate)[spectrum.argmax()]
    assert frequency == pytest.approx(220 * 2 ** (pitch / 12), abs=2)
    assert len(audio) / rate == pytest.approx(1.0, abs=.06)
    assert max(abs(audio)) < 1
    assert len(generated) == 1 and generated[0][1] == source_id
    mp3 = tts_delivery.encode_mp3(child, backend._audio_cache_dir())
    assert {name.split("/")[1] for name in files} == {"zerotts", "voice_color", "mp3"}
    # A rebuild must recover both voices and their encoded output with no
    # inference, pitch processing or MP3 encoding.
    monkeypatch.setenv("ZEROTTS_DATA_DIR", str(tmp_path / "rebuilt"))
    monkeypatch.setattr(ZeroTTSBackend, "_load_engine", lambda *a: pytest.fail("Must restore source"))
    monkeypatch.setattr(tts_delivery.subprocess, "run", lambda *a, **kw: pytest.fail("Must restore processed cache"))
    monkeypatch.setattr(tts_delivery, "_encode_mp3_data", lambda *a: pytest.fail("Must restore MP3"))
    monkeypatch.setattr(tts_delivery, "_MP3_CACHE", type(tts_delivery._MP3_CACHE)())
    restarted = ZeroTTSBackend()
    assert restarted.synthesize_wav("Xin chao", source_id) == source
    assert restarted.synthesize_wav("Xin chao", voice_id) == child
    assert tts_delivery.encode_mp3(child, restarted._audio_cache_dir()) == mp3


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_revised_processing_bypasses_old_rough_effect_cache(backend):
    source = tone()
    old_settings = b"4.000000:1.100000"
    old_key = hashlib.sha256(b"rubberband-voice-color-pcm16-v1\0" + old_settings + b"\0" + source).hexdigest()
    directory = backend._audio_cache_dir()
    old_path = directory / "voice_color" / (old_key + ".wav")
    old_path.parent.mkdir(parents=True)
    old_audio = tone(880)
    old_path.write_bytes(old_audio)
    revised = tts_delivery.change_wav_voice_color(source, directory, pitch_semitones=4, formant_ratio=1.10)
    assert revised != old_audio
    assert old_path.read_bytes() == old_audio
    assert len(list(old_path.parent.glob("*.wav"))) == 2


def test_failed_child_effect_keeps_original_for_retry(backend, monkeypatch):
    source = tone()
    calls = []
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda *a, **kw: calls.append(a) or source)

    def fail(*args, **kwargs):
        raise tts_delivery.AudioEncodingError("No rubberband")

    monkeypatch.setattr(tts_delivery, "change_wav_voice_color", fail)
    with pytest.raises(ZeroTTSBackendError, match="rubberband"):
        backend.synthesize_wav("Xin chao", CHILD)
    assert backend.synthesize_wav("Xin chao", SOURCE) == source
    assert len(calls) == 1


@pytest.mark.parametrize("endpoint", ["query", "json", "zerotts"])
@pytest.mark.parametrize("voice_id,source_id,pitch", [(CHILD, SOURCE, 3.0),
    ("quangminh_child", "quangminh", 5.0), ("tiendat_child", "tiendat", 4.0),
    ("quangvinh_child", "htSzITlVXGL3s4XwNqOv", 4.0)])
def test_child_voice_routes_through_existing_api(backend, monkeypatch, endpoint, voice_id, source_id, pitch):
    source, child = tone(), tone(280)
    monkeypatch.setattr(server, "FORCE_VOICE_ID", False)
    monkeypatch.setattr(server, "zerotts_backend", backend)
    voices, effects = [], []
    monkeypatch.setattr(backend, "_synthesize_uncached", lambda text, voice, **kw: voices.append(voice) or source)
    monkeypatch.setattr(tts_delivery, "change_wav_voice_color", lambda *a, **kw: effects.append(kw) or child)
    payload = {"text": "Xin chao", "voice_id": voice_id, "format": "wav"}
    client = TestClient(server.app)
    if endpoint == "query":
        result = client.get("/tts", params={**payload, "provider": "zerotts"})
    elif endpoint == "json":
        result = client.post("/tts", json={**payload, "provider": "zerotts"})
    else:
        result = client.post("/zerotts/synthesize", json=payload)
    assert result.status_code == 200
    assert result.content == child
    assert voices == [source_id]
    assert effects == [{"pitch_semitones": pitch, "formant_ratio": None}]


@pytest.mark.parametrize("voice_id", ["quangminh_child", "tiendat_child", "quangvinh_child"])
def test_boys_are_discoverable_and_their_metadata_is_windows_safe(voice_id):
    backend = ZeroTTSBackend()
    voice = next(item for item in backend.list_voices() if item["voice_id"] == voice_id)
    assert "male" in voice["tags"] and "childlike" in voice["tags"]
    path = module._DEFAULT_BUNDLED_VOICES_DIR / voice_id
    metadata = json.loads((path / "meta.json").read_bytes().decode("cp1252"))
    source_id, pitch = module._BOY_VOICES[voice_id]
    assert metadata["source_voice_id"] == source_id
    assert metadata["delivery"]["pitch_semitones"] == pitch
    assert hashlib.sha256((path / "voice.npz").read_bytes()).hexdigest() == metadata["source_embedding_sha256"]


def test_southern_boy_keeps_the_catalogued_source_embedding():
    root = module._DEFAULT_BUNDLED_VOICES_DIR
    source = root / "htSzITlVXGL3s4XwNqOv"
    child = root / "quangvinh_child"
    metadata = json.loads((source / "meta.json").read_bytes().decode("cp1252"))
    assert metadata["tags"] == ["male", "young", "southern"]
    assert (source / "voice.npz").read_bytes() == (child / "voice.npz").read_bytes()
    with np.load(source / "voice.npz", allow_pickle=False) as pack:
        assert pack["voice_emb"].shape == (1, int(pack["n_voice_queries"]), 768)
        assert pack["voice_emb"].dtype == np.float32
        assert np.isfinite(pack["voice_emb"]).all()
        assert pack["voice_emb"].tobytes() == (source / "voice.bin").read_bytes()
    assert "southern" in json.loads((child / "meta.json").read_text(encoding="ascii"))["tags"]
