"""FastAPI server wrapping edge-tts for use with Flutter or any HTTP client."""

import io

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import edge_tts

app = FastAPI(title="edge-tts API")


class TTSRequest(BaseModel):
    text: str
    voice: str = "vi-VN-HoaiMyNeural"
    rate: str = "+0%"
    volume: str = "+0%"
    pitch: str = "+0Hz"


@app.post("/tts", response_class=StreamingResponse)
async def text_to_speech(req: TTSRequest):
    """Convert text to speech and return MP3 audio."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    communicate = edge_tts.Communicate(
        text=req.text,
        voice=req.voice,
        rate=req.rate,
        volume=req.volume,
        pitch=req.pitch,
    )

    audio_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])

    if audio_buffer.tell() == 0:
        raise HTTPException(status_code=500, detail="No audio received from service")

    audio_buffer.seek(0)
    return StreamingResponse(audio_buffer, media_type="audio/mpeg")


@app.get("/voices")
async def list_voices(locale: str = Query(default=None, description="Filter by locale, e.g. vi-VN")):
    """List all available voices, optionally filtered by locale."""
    voices = await edge_tts.list_voices()
    if locale:
        voices = [v for v in voices if v["Locale"].lower().startswith(locale.lower())]
    return voices
