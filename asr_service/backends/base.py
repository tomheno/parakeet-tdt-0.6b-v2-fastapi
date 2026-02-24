"""Abstract ASR backend interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class TranscriptionResult:
    """Unified transcription result across all backends."""
    text: str
    language: str = ""
    duration: float = 0.0
    timestamps: Optional[dict] = None


class ASRBackend(ABC):
    """Abstract ASR backend. Each model family implements this."""

    @abstractmethod
    async def start(self, app) -> None:
        """Initialize model, load weights, warm up. Store state on app."""

    @abstractmethod
    async def stop(self) -> None:
        """Release GPU memory, stop background tasks."""

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        target_language: Optional[str] = None,
        timestamps: bool = False,
        beam_size: int = 0,
    ) -> TranscriptionResult:
        """Transcribe a single audio file from raw bytes.

        beam_size: 0 = greedy (default, fastest), >1 = beam search.
        """

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        """Stream transcription results. Yields {"delta": str, "finished": bool}."""

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Return list of supported language codes."""

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return model name, backend type, capabilities."""
