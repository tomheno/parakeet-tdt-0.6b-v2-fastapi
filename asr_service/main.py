"""Unified ASR service — supports Canary and Qwen3 backends.

Select backend via ASR_BACKEND env var: "canary" (default) or "qwen3".
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .config import ASR_BACKEND, logger


def _create_backend():
    """Instantiate the backend selected by ASR_BACKEND env var."""
    if ASR_BACKEND == "canary":
        try:
            from .backends.canary.backend import CanaryBackend
        except ImportError as e:
            raise ImportError(
                "Canary backend requires NeMo. Install with: pip install nemo_toolkit[asr]"
            ) from e
        return CanaryBackend()
    elif ASR_BACKEND == "qwen3":
        try:
            from .backends.qwen3.backend import Qwen3Backend
        except ImportError as e:
            raise ImportError(
                "Qwen3 backend requires vllm. Install with: pip install vllm qwen-asr"
            ) from e
        return Qwen3Backend()
    else:
        raise ValueError(f"Unknown ASR_BACKEND='{ASR_BACKEND}'. Use 'canary' or 'qwen3'.")


@asynccontextmanager
async def lifespan(app):
    """Start backend on startup, release on shutdown."""
    backend = _create_backend()
    logger.info("Starting ASR backend: %s", ASR_BACKEND)
    await backend.start(app)
    app.state.backend = backend
    logger.info("ASR backend ready: %s", ASR_BACKEND)
    try:
        yield
    finally:
        logger.info("Shutting down ASR backend")
        await backend.stop()


def create_app() -> FastAPI:
    application = FastAPI(
        title="Unified ASR Service",
        version="2.0.0",
        description=(
            "Speech-to-text service supporting multiple backends. "
            f"Current backend: {ASR_BACKEND}"
        ),
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Health & info (shared)
    # ------------------------------------------------------------------

    @application.get("/health", tags=["health"])
    async def health():
        return JSONResponse({"status": "ok", "backend": ASR_BACKEND})

    @application.get("/healthz", tags=["health"])
    async def healthz():
        return JSONResponse({"status": "ok"})

    @application.get("/languages", tags=["info"])
    async def list_languages():
        backend = application.state.backend
        return {"languages": backend.get_supported_languages()}

    @application.get("/v1/models", tags=["info"])
    async def list_models():
        backend = application.state.backend
        info = backend.get_model_info()
        return {"data": [info]}

    # ------------------------------------------------------------------
    # OpenAI-compatible transcription endpoint (both backends)
    #
    # Accepts TWO content types for the same URL:
    #   1. multipart/form-data  — standard OpenAI API (file + form fields)
    #   2. audio/* raw body     — zero-overhead fast path (query params)
    #
    # Raw binary path skips multipart parsing entirely, giving the same
    # throughput as the old /transcribe/raw endpoint.
    # ------------------------------------------------------------------

    @application.post("/v1/audio/transcriptions", tags=["transcription"])
    async def transcribe_openai(request: Request):
        """OpenAI-compatible audio transcription endpoint.

        Accepts either:
          - multipart/form-data with `file`, `model`, `language`, `response_format`, `stream`
          - raw audio body with Content-Type: audio/* (params via query string)
        """
        content_type = request.headers.get("content-type", "")

        if "multipart/form-data" in content_type:
            # Standard OpenAI multipart path
            form = await request.form()
            file_field = form.get("file")
            if file_field is None:
                raise HTTPException(400, "Missing 'file' field")
            audio_bytes = await file_field.read()
            language = form.get("language")
            response_format = form.get("response_format", "json")
            stream = form.get("stream")
            beam_size = int(form.get("beam_size", "0"))
        else:
            # Raw binary fast path — audio bytes in body, params in query
            audio_bytes = await request.body()
            if not audio_bytes:
                raise HTTPException(400, "Empty audio")

            # Fast path: raw binary → direct batcher call (skip backend abstraction)
            # Supports language + beam_size params via query string
            batcher = getattr(application.state, "batcher", None)
            if batcher is not None:
                language = request.query_params.get("language", "en")
                beam_size = int(request.query_params.get("beam_size", "0"))
                output = await batcher.transcribe(
                    audio_bytes, source_lang=language, target_lang=language,
                    timestamps=False, beam_size=beam_size,
                )
                text = output[0].text if hasattr(output[0], "text") else str(output[0])
                return {"text": text.strip()}

            # Fallback for backends without direct batcher access
            query = request.query_params
            language = query.get("language")
            response_format = query.get("response_format", "json")
            stream = query.get("stream")
            beam_size = int(query.get("beam_size", "0"))

        if not audio_bytes:
            raise HTTPException(400, "Empty audio")

        is_stream = stream and str(stream).lower() == "true"

        if is_stream:
            backend = application.state.backend
            async def sse():
                async for chunk in backend.transcribe_stream(audio_bytes, language=language):
                    data = json.dumps({
                        "choices": [{
                            "delta": {"content": chunk["delta"]},
                            "finish_reason": "stop" if chunk["finished"] else None,
                        }]
                    })
                    yield f"data: {data}\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        backend = application.state.backend
        result = await backend.transcribe(audio_bytes, language=language, beam_size=beam_size)

        fmt = (response_format or "json").lower()
        if fmt == "text":
            return PlainTextResponse(result.text)
        if fmt == "verbose_json":
            return {
                "task": "transcribe",
                "language": result.language or "unknown",
                "duration": result.duration,
                "text": result.text,
            }
        return {"text": result.text}

    # ------------------------------------------------------------------
    # A/B test: old-style /transcribe/raw (bypasses backend abstraction)
    # ------------------------------------------------------------------

    @application.post("/transcribe/raw", tags=["transcription"])
    async def transcribe_raw_direct(request: Request):
        """Direct batcher access — identical to old canary_service endpoint."""
        content = await request.body()
        beam_size = int(request.query_params.get("beam_size", "0"))
        batcher = application.state.batcher
        output = await batcher.transcribe(
            content,
            source_lang="en",
            target_lang="en",
            timestamps=False,
            beam_size=beam_size,
        )
        text = output[0].text if hasattr(output[0], "text") else str(output[0])
        return {"text": text.strip()}

    logger.info("FastAPI app initialized — backend=%s", ASR_BACKEND)
    return application


app = create_app()
