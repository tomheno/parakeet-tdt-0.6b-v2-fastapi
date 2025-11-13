from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from .model import lifespan
from .config import logger

from parakeet_service.streaming_server import websocket_streaming_endpoint

def create_app() -> FastAPI:
    app = FastAPI(
        title="Parakeet TDT Streaming STT for LiveKit",
        version="1.0.0",
        description=(
            "Low-latency streaming speech-to-text using NVIDIA Parakeet-TDT "
            "with cache-aware inference for LiveKit agents."
        ),
        lifespan=lifespan,
    )

    # Health check endpoint
    @app.get("/healthz", tags=["health"])
    async def health_check():
        """Health check endpoint for monitoring and load balancers."""
        return JSONResponse({"status": "ok"})

    # Streaming endpoint (cache-aware, low-latency for LiveKit)
    @app.websocket("/stream")
    async def stream_endpoint(websocket: WebSocket):
        """
        Low-latency streaming STT endpoint using cache-aware inference.

        Protocol:
        - Client sends: PCM int16 audio bytes (16kHz mono)
        - Server sends: JSON {"text": "...", "is_final": false, "session_id": "..."}
        """
        model = app.state.asr_model
        await websocket_streaming_endpoint(websocket, model)

    logger.info("FastAPI app initialised with streaming endpoint for LiveKit")
    return app


app = create_app()
