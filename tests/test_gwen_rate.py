"""Gwen rate preserves source audio, pitch, delivery semantics and other engines."""

from io import BytesIO
import os
import shutil
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
import tts_delivery as delivery
from gwen_backend import GwenBackend


def tone():
    output = BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24000)
        samples = (np.sin(np.arange(72000) * (2 * np.pi * 220 / 24000)) * 12000).astype("<i2")
        target.writeframes(samples.tobytes())
    return output.getvalue()


def samples(audio):
    with wave.open(BytesIO(audio), "rb") as source:
        rate = source.getframerate()
        data = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(float)
    return data, rate


@pytest.fixture
def backend(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "FORCE_VOICE_ID", False)
    monkeypatch.setenv("GWEN_AUDIO_CACHE_DIR", str(tmp_path))
    result = GwenBackend()
    monkeypatch.setattr(server, "gwen_backend", result)
    calls = []

    def generate(text, voice_id, path):
        calls.append(text)
        path.write_bytes(tone())

    monkeypatch.setattr(result, "_run_worker", generate)
    yield result, calls
    result.close()


@pytest.mark.parametrize("method", ["get", "post"])
def test_omitted_or_zero_rate_returns_original_bytes(backend, monkeypatch, method):
    def fail(*args):
        pytest.fail("Default and zero rate must not process audio")

    monkeypatch.setattr(server, "change_wav_tempo", fail)
    client = TestClient(server.app)
    payload = {"text": "Hello", "provider": "gwen", "voice": "yen_nhi"}
    for extra in ({}, {"rate": "+0%"}):
        response = getattr(client, method)("/tts", **{"json" if method == "post" else "params": payload | extra})
        assert response.status_code == 200 and response.content == tone()
    assert backend[1] == ["Hello"]


@pytest.mark.parametrize("rate", ["-100%", "-51%", "+101%", "slow", "0.8", "nan", "-20%;volume=0"])
def test_bad_rate_rejected_before_inference(backend, rate):
    response = TestClient(server.app).get("/tts", params={"text": "Hello", "provider": "gwen", "rate": rate})
    assert response.status_code == 400 and "Gwen rate" in response.json()["detail"]
    assert backend[1] == []


def test_missing_ffmpeg_fails_clearly_but_original_still_works(backend, monkeypatch):
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    monkeypatch.setattr(delivery.shutil, "which", lambda _: None)
    client = TestClient(server.app)
    payload = {"text": "Hello", "provider": "gwen"}
    response = client.get("/tts", params=payload | {"rate": "-20%"})
    assert response.status_code == 503 and "ffmpeg" in response.json()["detail"]
    assert client.get("/tts", params=payload).content == tone()
    assert backend[1] == ["Hello"]


@pytest.mark.skipif(not (os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")), reason="ffmpeg required")
@pytest.mark.parametrize("method", ["get", "post"])
def test_real_atempo_preserves_pitch_caches_and_reuses_source(backend, monkeypatch, tmp_path, method):
    original = tone()
    client = TestClient(server.app)
    payload = {"text": "Hello", "provider": "gwen", "rate": "-20%"}
    response = getattr(client, method)("/tts", **{"json" if method == "post" else "params": payload})
    assert response.status_code == 200, response.text
    signal, sr = samples(response.content)
    original_signal, original_sr = samples(original)
    assert sr == original_sr == 24000
    assert len(signal) / len(original_signal) == pytest.approx(1.25, abs=.03)
    spectrum = abs(np.fft.rfft(signal * np.hanning(len(signal))))
    peak_hz = np.fft.rfftfreq(len(signal), 1 / sr)[spectrum.argmax()]
    assert peak_hz == pytest.approx(220, abs=1)
    assert backend[1] == ["Hello"]
    assert next(tmp_path.glob("*.wav")).read_bytes() == original
    assert len(list((tmp_path / "tempo").glob("*.wav"))) == 1

    def fail(*args, **kwargs):
        pytest.fail("A cached tempo must not run ffmpeg again")

    monkeypatch.setattr(delivery.subprocess, "run", fail)
    assert client.get("/tts", params=payload).content == response.content
    part = client.get("/tts", params=payload, headers={"Range": "bytes=0-19"})
    assert part.status_code == 206 and part.content == response.content[:20]
    head = client.head("/tts", params=payload)
    assert head.headers["etag"] == response.headers["etag"]
    assert int(head.headers["content-length"]) == len(response.content)
    assert client.get("/tts", params=payload, headers={"If-None-Match": response.headers["etag"]}).status_code == 304
    assert client.get("/tts", params={"text": "Hello", "provider": "gwen"}).content == original


def test_mp3_encoding_receives_processed_audio(backend, monkeypatch):
    captured = []
    monkeypatch.setattr(server, "change_wav_tempo", lambda *args: b"processed-wav")

    def encode(audio, cache_dir):
        captured.append(audio)
        return b"ID3-processed"

    monkeypatch.setattr(server, "encode_mp3", encode)
    response = TestClient(server.app).get("/tts", params={
        "text": "Hello", "provider": "gwen", "rate": "-20%", "format": "mp3"})
    assert response.status_code == 200 and response.content == b"ID3-processed"
    assert captured == [b"processed-wav"]
