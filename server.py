"""FastAPI server wrapping edge-tts for use with Flutter or any HTTP client."""

from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# To run the server, use the following command:
# uvicorn server:app --reload --host 127.0.0.1 --port 8000

import edge_tts

app = FastAPI(title="edge-tts API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str
    voice: str = "vi-VN-HoaiMyNeural"
    rate: str = "-25%"
    volume: str = "+100%"
    pitch: str = "+0Hz"


def _build_tts_response(
    *,
    text: str,
    voice: str,
    rate: str,
    volume: str,
    pitch: str,
) -> StreamingResponse:
    """Create a streaming MP3 response with minimal buffering."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        volume=volume,
        pitch=pitch,
    )

    async def audio_chunks() -> AsyncGenerator[bytes, None]:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    return StreamingResponse(
        audio_chunks(),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "inline; filename=tts.mp3",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Convert text to speech and stream MP3 audio."""
    return _build_tts_response(
        text=req.text,
        voice=req.voice,
        rate=req.rate,
        volume=req.volume,
        pitch=req.pitch,
    )


@app.get("/tts")
async def text_to_speech_get(
    text: str = Query(description="Text to synthesize"),
    voice: str = Query(default="vi-VN-HoaiMyNeural"),
    rate: str = Query(default="-10%"),
    volume: str = Query(default="+100%"),
    pitch: str = Query(default="+0Hz"),
):
    """Convert text to speech and stream MP3 audio from query params."""
    return _build_tts_response(
        text=text,
        voice=voice,
        rate=rate,
        volume=volume,
        pitch=pitch,
    )


@app.get("/voices")
async def list_voices(locale: str = Query(default=None, description="Filter by locale, e.g. vi-VN")):
    """List all available voices, optionally filtered by locale."""
    voices = await edge_tts.list_voices()
    if locale:
        voices = [v for v in voices if v["Locale"].lower().startswith(locale.lower())]
    return voices
