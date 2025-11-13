"""
LiveKit STT plugin for Parakeet TDT streaming inference.

Usage:
    from livekit_plugin import ParakeetSTT

    stt = ParakeetSTT()
    async with stt.stream() as stream:
        # Feed audio frames
        stream.push_frame(audio_frame)

        # Receive transcripts
        async for event in stream:
            if event.type == SpeechEventType.INTERIM_TRANSCRIPT:
                print(event.alternatives[0].text)
"""
from __future__ import annotations
import asyncio
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
import logging

# LiveKit imports
try:
    from livekit.agents import stt, utils
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    logging.warning("LiveKit not installed. Plugin will not work.")

# NeMo imports
import nemo.collections.asr as nemo_asr

# NeMo streaming
try:
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    logging.warning("NeMo streaming not available")

logger = logging.getLogger(__name__)


@dataclass
class ParakeetOptions:
    """Configuration for Parakeet STT."""
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3"
    chunk_size: float = 0.08  # 80ms
    shift_size: float = 0.04  # 40ms overlap
    left_chunks: int = 4      # context chunks
    language: str = "en"
    device: str = "cuda"
    precision: str = "fp16"   # fp16 or fp32


if LIVEKIT_AVAILABLE:
    class ParakeetSTT(stt.STT):
        """
        Parakeet TDT streaming STT for LiveKit agents.

        Provides low-latency (<300ms) speech-to-text using NVIDIA's Parakeet-TDT model.
        """

        def __init__(
            self,
            *,
            model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
            chunk_size: float = 0.08,
            shift_size: float = 0.04,
            left_chunks: int = 4,
            language: str = "en",
            device: str = "cuda",
            precision: str = "fp16"
        ):
            super().__init__(
                capabilities=stt.STTCapabilities(
                    streaming=True,
                    interim_results=True
                )
            )

            self._opts = ParakeetOptions(
                model_name=model_name,
                chunk_size=chunk_size,
                shift_size=shift_size,
                left_chunks=left_chunks,
                language=language,
                device=device,
                precision=precision
            )

            # Load model
            logger.info(f"Loading Parakeet model: {model_name}")
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name,
                map_location=device
            )

            # Set precision
            if precision == "fp16" and torch.cuda.is_available():
                self._model = self._model.half()
                logger.info("Using FP16 precision")

            self._model.eval()

            # Enable streaming attention if available
            try:
                self._model.change_attention_model(
                    self_attention_model="rel_pos_local_attn",
                    att_context_size=[256, 64]
                )
                logger.info("Enabled cache-aware attention")
            except Exception as e:
                logger.warning(f"Could not enable cache-aware attention: {e}")

            logger.info("Parakeet STT initialized")

        def stream(
            self,
            *,
            language: Optional[str] = None
        ) -> ParakeetStream:
            """Create a new streaming session."""
            return ParakeetStream(
                stt=self,
                opts=self._opts,
                model=self._model
            )


    class ParakeetStream(stt.SpeechStream):
        """Streaming session for Parakeet STT."""

        def __init__(
            self,
            *,
            stt: ParakeetSTT,
            opts: ParakeetOptions,
            model
        ):
            super().__init__(stt=stt, sample_rate=16000)

            self._opts = opts
            self._model = model
            self._buffer: Optional[CacheAwareStreamingAudioBuffer] = None
            self._speaking = False
            self._last_transcript = ""

        async def _run(self):
            """Main processing loop."""
            # Initialize streaming buffer
            if STREAMING_AVAILABLE:
                self._buffer = CacheAwareStreamingAudioBuffer(
                    model=self._model,
                    chunk_size=self._opts.chunk_size,
                    shift_size=self._opts.shift_size,
                    left_chunks=self._opts.left_chunks,
                    online_normalization=True
                )
                logger.info("Using CacheAwareStreamingAudioBuffer")
            else:
                logger.warning("Streaming buffer not available, using fallback")
                self._audio_buffer = []

            try:
                async for frame in self._input_ch:
                    if isinstance(frame, rtc.AudioFrame):
                        await self._process_frame(frame)
                    elif isinstance(frame, self._FlushSentinel):
                        # End of speech - send final transcript
                        if self._speaking:
                            await self._emit_final()

            except Exception as e:
                logger.error(f"Error in streaming: {e}", exc_info=True)
            finally:
                # Cleanup
                if self._buffer:
                    self._buffer.reset()

        async def _process_frame(self, frame: rtc.AudioFrame):
            """Process a single audio frame."""
            # Convert to numpy array
            audio_data = np.frombuffer(
                frame.data.tobytes(),
                dtype=np.int16
            ).astype(np.float32) / 32768.0

            # Process through streaming buffer
            if self._buffer:
                try:
                    with torch.inference_mode():
                        transcript = self._buffer.infer_signal(audio_data)

                    if transcript and transcript != self._last_transcript:
                        # Start of speech event
                        if not self._speaking:
                            self._speaking = True
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=stt.SpeechEventType.START_OF_SPEECH
                                )
                            )

                        # Interim transcript
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                alternatives=[
                                    stt.SpeechData(
                                        text=transcript,
                                        language=self._opts.language
                                    )
                                ]
                            )
                        )
                        self._last_transcript = transcript

                except Exception as e:
                    logger.error(f"Inference error: {e}")
            else:
                # Fallback: accumulate audio
                self._audio_buffer.append(audio_data)

        async def _emit_final(self):
            """Emit final transcript event."""
            if self._last_transcript:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[
                            stt.SpeechData(
                                text=self._last_transcript,
                                language=self._opts.language
                            )
                        ]
                    )
                )

            # End of speech
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH
                )
            )

            # Reset state
            self._speaking = False
            self._last_transcript = ""
            if self._buffer:
                self._buffer.reset()

else:
    # Stub classes if LiveKit not available
    class ParakeetSTT:
        def __init__(self, **kwargs):
            raise ImportError("LiveKit SDK not installed. Install with: pip install livekit-agents")

    class ParakeetStream:
        pass
