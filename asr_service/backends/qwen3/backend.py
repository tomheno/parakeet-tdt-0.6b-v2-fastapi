"""Qwen3-ASR backend adapter — wraps ASREngine (vLLM AsyncLLM)."""

import logging
from typing import AsyncIterator, Optional

from ..base import ASRBackend, TranscriptionResult
from .config import (
    ENABLE_PREFIX_CACHING,
    GPU_MEMORY_UTILIZATION,
    MAX_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    MM_PROCESSOR_CACHE_GB,
    QWEN3_MODEL,
)

logger = logging.getLogger("asr_service.qwen3")

# Qwen3-ASR supported languages (52 languages + 22 Chinese dialects)
SUPPORTED_LANGUAGES = [
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo",
    "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es",
    "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw",
    "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja",
    "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo",
    "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
    "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt",
    "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
    "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
    "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "yue", "zh",
]


class Qwen3Backend(ASRBackend):
    """Qwen3-ASR backend using vLLM AsyncLLM engine."""

    def __init__(self):
        self.engine = None

    async def start(self, app) -> None:
        try:
            from .engine import ASREngine
        except ImportError as e:
            raise ImportError(
                "Qwen3 backend requires vllm and qwen_asr packages. "
                "Install with: pip install vllm qwen-asr"
            ) from e

        self.engine = ASREngine(
            model=QWEN3_MODEL,
            gpu_memory=GPU_MEMORY_UTILIZATION,
            max_seqs=MAX_NUM_SEQS,
            max_batched_tokens=MAX_BATCHED_TOKENS,
            enable_prefix_caching=ENABLE_PREFIX_CACHING,
            mm_processor_cache_gb=MM_PROCESSOR_CACHE_GB,
        )
        await self.engine.start()

    async def stop(self) -> None:
        self.engine = None
        logger.info("Qwen3 backend stopped")

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        target_language: Optional[str] = None,
        timestamps: bool = False,
    ) -> TranscriptionResult:
        from .engine import decode_audio

        audio = decode_audio(audio_bytes)
        result = await self.engine.transcribe(audio, language=language)

        return TranscriptionResult(
            text=result["text"],
            language=result.get("language", language or ""),
            duration=result.get("duration", 0.0),
        )

    async def transcribe_stream(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        from .engine import decode_audio

        audio = decode_audio(audio_bytes)
        async for chunk in self.engine.transcribe_stream(audio, language=language):
            yield chunk

    def get_supported_languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)

    def get_model_info(self) -> dict:
        return {
            "id": "qwen3-asr",
            "name": f"Qwen3-ASR ({QWEN3_MODEL})",
            "backend": "vllm",
            "languages": len(SUPPORTED_LANGUAGES),
            "capabilities": ["transcription"],
        }
