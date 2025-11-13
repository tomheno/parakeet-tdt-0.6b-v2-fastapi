from contextlib import asynccontextmanager
import gc
import torch
import nemo.collections.asr as nemo_asr

from .config import MODEL_NAME, MODEL_PRECISION, DEVICE, logger


@asynccontextmanager
async def lifespan(app):
    """Load model once per process; free GPU on shutdown."""
    logger.info("Loading %s with optimized memory...", MODEL_NAME)
    with torch.inference_mode():
        # Determine precision
        dtype = torch.float16 if MODEL_PRECISION == "fp16" else torch.float32

        # Load model with configurable device and precision
        model = nemo_asr.models.ASRModel.from_pretrained(
            MODEL_NAME,
            map_location=DEVICE
        ).to(dtype=dtype)
        logger.info("Loaded model with %s weights on %s", MODEL_PRECISION.upper(), DEVICE)

    # Configure for streaming (cache-aware attention)
    try:
        from parakeet_service.streaming_server import setup_model_for_streaming
        model = setup_model_for_streaming(model)
    except Exception as e:
        logger.warning("Could not configure streaming mode: %s", e)

    # Aggressive cleanup
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Memory cleanup complete")

    app.state.asr_model = model
    logger.info("Model ready on %s (streaming mode for LiveKit)", next(model.parameters()).device)

    try:
        yield
    finally:
        logger.info("Releasing GPU memory and shutting down")
        del app.state.asr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # free cache but keep driver
