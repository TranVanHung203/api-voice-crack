"""Duration describes delivered audio and remains readable by browser clients."""

from io import BytesIO
import shutil

from fastapi import Request
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

import server
import tts_delivery as delivery


@pytest.fixture(params=["wav", "mp3"])
def audio_file(request):
    rate = 48000
    samples = .1 * np.sin(2 * np.pi * 220 * np.arange(59376) / rate)
    output = BytesIO()
    sf.write(output, samples, rate, format="WAV", subtype="PCM_16")
    audio = output.getvalue()
    if request.param == "mp3":
        if not shutil.which("ffmpeg"):
            pytest.importorskip("av")
        audio = delivery._encode_mp3_data(audio)
    return request.param, audio, len(samples) / rate


def test_duration_describes_full_file_for_ranges_head_and_revalidation(audio_file):
    extension, audio, seconds = audio_file
    media_type = "audio/mpeg" if extension == "mp3" else "audio/wav"

    def response(method="GET", headers=()):
        request = Request({"type": "http", "method": method, "headers": headers})
        return delivery.audio_response(audio, media_type, "test." + extension, request)

    full = response()
    duration = full.headers["x-audio-duration-seconds"]
    assert float(duration) == pytest.approx(seconds, abs=.001)
    assert full.body == audio
    partial = response(headers=[(b"range", b"bytes=0-30")])
    assert partial.status_code == 206 and partial.body == audio[:31]
    head = response("HEAD")
    assert head.body == b"" and int(head.headers["content-length"]) == len(audio)
    unchanged = response(headers=[(b"if-none-match", full.headers["etag"].encode())])
    assert unchanged.status_code == 304 and unchanged.body == b""
    for result in (partial, head, unchanged):
        assert result.headers["x-audio-duration-seconds"] == duration


@pytest.mark.parametrize("provider,path", [
    ("zerotts", "/tts"), ("vieneu", "/tts"), ("gwen", "/tts"),
    ("zerotts", "/zerotts/synthesize"),
])
def test_duration_reaches_frontend_for_each_supported_provider(monkeypatch, audio_file, provider, path):
    extension, audio, seconds = audio_file
    monkeypatch.setattr(server, "FORCE_VOICE_ID", False)
    backend = getattr(server, provider + "_backend")
    monkeypatch.setattr(backend, "synthesize_wav", lambda *a, **kw: audio)
    monkeypatch.setattr(server, "encode_mp3", lambda *a, **kw: audio)
    client = TestClient(server.app)
    payload = {"text": "Xin chao", "provider": provider, "voice_id": "hamy", "format": extension}
    result = client.post(path, json=payload, headers={"Origin": "https://frontend.example"})
    assert result.status_code == 200 and result.content == audio
    assert float(result.headers["x-audio-duration-seconds"]) == pytest.approx(seconds, abs=.001)
    exposed = result.headers["access-control-expose-headers"].lower().split(", ")
    assert "x-audio-duration-seconds" in exposed


def test_unreadable_metadata_does_not_break_audio_delivery():
    audio = b"ID3-legacy-unreadable-metadata"
    result = delivery.audio_response(audio, "audio/mpeg", "legacy.mp3", None)
    assert result.status_code == 200 and result.body == audio
    assert "x-audio-duration-seconds" not in result.headers
