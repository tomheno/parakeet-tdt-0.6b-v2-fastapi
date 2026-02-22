import asyncio
from typing import List

import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse

from .config import SUPPORTED_LANGUAGES, TARGET_SR, logger
from .model import lifespan
from .streaming_server import websocket_streaming_endpoint


def create_app() -> FastAPI:
    application = FastAPI(
        title="Canary 1B V2 — Multilingual STT & Translation",
        version="1.0.0",
        description=(
            "Speech-to-text and speech translation using NVIDIA Canary 1B V2. "
            "Supports 25 European languages with per-request language selection. "
            "Set source_lang == target_lang for ASR, or different for translation."
        ),
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Health & info
    # ------------------------------------------------------------------

    @application.get("/healthz", tags=["health"])
    async def health_check():
        """Liveness probe."""
        return JSONResponse({"status": "ok"})

    @application.get("/languages", tags=["info"])
    async def list_languages():
        """Return the list of supported language codes."""
        return {
            "languages": SUPPORTED_LANGUAGES,
            "note": (
                "Use the same code for source_lang and target_lang to transcribe. "
                "Use different codes to translate (e.g. source_lang=fr, target_lang=en)."
            ),
        }

    # ------------------------------------------------------------------
    # REST transcription / translation
    # ------------------------------------------------------------------

    @application.post("/transcribe", tags=["transcription"])
    async def transcribe(
        audio: UploadFile = File(..., description="Audio file (WAV, FLAC, MP3 …)"),
        source_lang: str = Form("en", description="Source language code"),
        target_lang: str = Form(
            "en",
            description="Target language code (same as source = ASR, different = translate)",
        ),
        timestamps: bool = Form(False, description="Return word/segment timestamps"),
    ):
        """
        Transcribe or translate an uploaded audio file.

        - **ASR**: ``source_lang=en, target_lang=en``
        - **Translation** (X → En): ``source_lang=fr, target_lang=en``
        - **Translation** (En → X): ``source_lang=en, target_lang=fr``
        """
        if source_lang not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                400,
                f"Unsupported source_lang '{source_lang}'. Supported: {SUPPORTED_LANGUAGES}",
            )
        if target_lang not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                400,
                f"Unsupported target_lang '{target_lang}'. Supported: {SUPPORTED_LANGUAGES}",
            )

        content = await audio.read()

        batcher = application.state.batcher
        output = await batcher.transcribe(
            content,                      # pass bytes directly — no temp file
            source_lang=source_lang,
            target_lang=target_lang,
            timestamps=timestamps,
        )

        text = output[0].text if hasattr(output[0], "text") else str(output[0])
        result = {"text": text.strip()}

        if timestamps and hasattr(output[0], "timestamp") and output[0].timestamp:
            result["timestamps"] = output[0].timestamp

        return result

    # ------------------------------------------------------------------
    # Raw binary transcription (no multipart overhead)
    # ------------------------------------------------------------------

    @application.post("/transcribe/raw", tags=["transcription"])
    async def transcribe_raw(
        request: Request,
        source_lang: str = Query("en", description="Source language code"),
        target_lang: str = Query(
            "en",
            description="Target language code (same = ASR, different = translate)",
        ),
        timestamps: bool = Query(False, description="Return word/segment timestamps"),
    ):
        """
        Low-overhead transcription: send raw audio bytes as request body.

        Bypasses multipart form parsing for ~3x lower per-request overhead.
        Use Content-Type: audio/wav (or any format soundfile supports).
        """
        if source_lang not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                400,
                f"Unsupported source_lang '{source_lang}'. Supported: {SUPPORTED_LANGUAGES}",
            )
        if target_lang not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                400,
                f"Unsupported target_lang '{target_lang}'. Supported: {SUPPORTED_LANGUAGES}",
            )

        content = await request.body()

        batcher = application.state.batcher
        output = await batcher.transcribe(
            content,
            source_lang=source_lang,
            target_lang=target_lang,
            timestamps=timestamps,
        )

        text = output[0].text if hasattr(output[0], "text") else str(output[0])
        result = {"text": text.strip()}

        if timestamps and hasattr(output[0], "timestamp") and output[0].timestamp:
            result["timestamps"] = output[0].timestamp

        return result

    # ------------------------------------------------------------------
    # Batch transcription
    # ------------------------------------------------------------------

    @application.post("/transcribe/batch", tags=["transcription"])
    async def transcribe_batch(
        audio: List[UploadFile] = File(
            ..., description="One or more audio files"
        ),
        source_lang: str = Form("en"),
        target_lang: str = Form("en"),
    ):
        """
        Transcribe multiple audio files in a single request.

        All files share the same language pair.  They are submitted to
        the in-flight batcher concurrently and processed in the same
        GPU batch when possible.

        Returns ``{"results": [{"text": "..."}, ...]}`` in the same
        order as the uploaded files.
        """
        if source_lang not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                400,
                f"Unsupported source_lang '{source_lang}'. "
                f"Supported: {SUPPORTED_LANGUAGES}",
            )
        if target_lang not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                400,
                f"Unsupported target_lang '{target_lang}'. "
                f"Supported: {SUPPORTED_LANGUAGES}",
            )

        batcher = application.state.batcher

        # Read all files and submit concurrently
        async def process_one(f: UploadFile):
            content = await f.read()
            output = await batcher.transcribe(
                content,
                source_lang=source_lang,
                target_lang=target_lang,
                timestamps=False,
            )
            text = output[0].text if hasattr(output[0], "text") else str(output[0])
            return {"text": text.strip()}

        results = await asyncio.gather(*[process_one(f) for f in audio])
        return {"results": list(results)}

    # ------------------------------------------------------------------
    # WebSocket streaming
    # ------------------------------------------------------------------

    @application.websocket("/stream")
    async def stream_endpoint(websocket: WebSocket):
        """
        Low-latency streaming STT / translation over WebSocket.

        1. Connect to ``/stream``
        2. Send JSON config: ``{"source_lang": "en", "target_lang": "en"}``
        3. Stream PCM int16 audio bytes (16 kHz mono)
        4. Receive JSON transcripts: ``{"text": "…", "is_final": false, …}``
        """
        model = application.state.asr_model
        batcher = application.state.batcher
        await websocket_streaming_endpoint(websocket, model, batcher)

    logger.info("FastAPI app initialised — Canary 1B V2 (ASR + translation, 25 languages)")
    return application


app = create_app()
