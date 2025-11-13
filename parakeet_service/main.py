from fastapi import FastAPI, WebSocket

from .model import lifespan
from .routes import router
from .config import logger

from parakeet_service.stream_routes import router as stream_router
from parakeet_service.streaming_server import websocket_streaming_endpoint, setup_model_for_streaming

def create_app() -> FastAPI:
    app = FastAPI(
        title="Parakeet-TDT 0.6B v2 STT service",
        version="0.0.1",
        description=(
            "High-accuracy English speech-to-text (FastConformer-TDT) "
            "with optional word/char/segment timestamps."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)

    # Legacy batch streaming endpoint (VAD-based)
    app.include_router(stream_router)

    # New: True streaming endpoint (cache-aware, low-latency)
    @app.websocket("/stream")
    async def stream_endpoint(websocket: WebSocket):
        """Low-latency streaming STT endpoint using cache-aware inference."""
        model = app.state.asr_model
        await websocket_streaming_endpoint(websocket, model)

    logger.info("FastAPI app initialised with streaming endpoints")
    return app


app = create_app()
