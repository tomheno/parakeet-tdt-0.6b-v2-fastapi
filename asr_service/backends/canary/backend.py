"""Canary ASR backend adapter — wraps existing direct_batcher + optimizations."""

from typing import AsyncIterator, Optional

from ..base import ASRBackend, TranscriptionResult
from .config import SUPPORTED_LANGUAGES, logger


class CanaryBackend(ASRBackend):
    """Canary 1B V2 backend using NeMo direct inference batcher."""

    def __init__(self):
        self.model = None
        self.batcher = None

    async def start(self, app) -> None:
        from .model import _start_canary

        model, batcher = await _start_canary()
        self.model = model
        self.batcher = batcher
        # Also store on app.state for native endpoints
        app.state.asr_model = model
        app.state.batcher = batcher

    async def stop(self) -> None:
        import gc
        import torch

        if self.batcher:
            self.batcher.stop()
        self.batcher = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Canary backend stopped")

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        target_language: Optional[str] = None,
        timestamps: bool = False,
        beam_size: int = 0,
    ) -> TranscriptionResult:
        source_lang = language or "en"
        target_lang = target_language or source_lang

        output = await self.batcher.transcribe(
            audio_bytes,
            source_lang=source_lang,
            target_lang=target_lang,
            timestamps=timestamps,
            beam_size=beam_size,
        )

        hyp = output[0]
        text = hyp.text if hasattr(hyp, "text") else str(hyp)
        ts = getattr(hyp, "timestamp", None) if timestamps else None

        return TranscriptionResult(
            text=text.strip(),
            language=source_lang,
            timestamps=ts,
        )

    async def transcribe_stream(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        # Canary uses buffered streaming (accumulate audio, run full transcribe)
        # For the OpenAI SSE interface, just return the full result as a single chunk
        result = await self.transcribe(audio_bytes, language=language)
        yield {"delta": result.text, "finished": True}

    def get_supported_languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)

    def get_model_info(self) -> dict:
        return {
            "id": "canary-1b-v2",
            "name": "NVIDIA Canary 1B V2",
            "backend": "nemo-direct-batcher",
            "languages": len(SUPPORTED_LANGUAGES),
            "capabilities": ["transcription", "translation"],
        }
