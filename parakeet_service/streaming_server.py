"""
True streaming STT server using NeMo's cache-aware streaming inference.
Low-latency alternative to batch processing.
"""
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
import torch
import asyncio
import uuid
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# NeMo streaming imports
try:
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    STREAMING_AVAILABLE = True
except ImportError:
    logger.warning("NeMo streaming utils not available, falling back to chunk-based inference")
    STREAMING_AVAILABLE = False


class StreamingSession:
    """Manages a single streaming STT session with cache-aware inference."""

    def __init__(self, model, session_id: str):
        self.session_id = session_id
        self.model = model

        if STREAMING_AVAILABLE:
            # Use NeMo's cache-aware streaming buffer
            self.buffer = CacheAwareStreamingAudioBuffer(
                model=model,
                chunk_size=0.08,          # 80ms chunks
                shift_size=0.04,          # 40ms overlap for smoothness
                left_chunks=4,            # 320ms historical context
                online_normalization=True
            )
            logger.info(f"Session {session_id}: Using CacheAwareStreamingAudioBuffer")
        else:
            # Fallback: simple chunked inference
            self.buffer = None
            self.audio_buffer = []
            logger.info(f"Session {session_id}: Using fallback chunked inference")

    def process_audio(self, audio_chunk: np.ndarray) -> str:
        """Process audio chunk and return transcript (if any)."""
        if STREAMING_AVAILABLE and self.buffer:
            try:
                result = self.buffer.infer_signal(audio_chunk)
                return result if result else ""
            except Exception as e:
                logger.error(f"Streaming inference error: {e}")
                return ""
        else:
            # Fallback: accumulate and transcribe every N chunks
            self.audio_buffer.append(audio_chunk)
            if len(self.audio_buffer) >= 10:  # ~800ms worth
                audio = np.concatenate(self.audio_buffer)
                self.audio_buffer = []

                # Use model's offline transcribe on accumulated audio
                import tempfile
                import soundfile as sf

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, audio, 16000)
                    try:
                        with torch.inference_mode():
                            result = self.model.transcribe([tmp.name], batch_size=1)
                        return result[0] if result else ""
                    except Exception as e:
                        logger.error(f"Fallback inference error: {e}")
                        return ""
            return ""

    def reset(self):
        """Reset session state."""
        if STREAMING_AVAILABLE and self.buffer:
            self.buffer.reset()
        else:
            self.audio_buffer = []


# Global session storage
streaming_sessions: Dict[str, StreamingSession] = {}


async def websocket_streaming_endpoint(ws: WebSocket, model):
    """
    WebSocket endpoint for real-time streaming STT.

    Protocol:
    - Client sends: PCM int16 audio bytes (16kHz mono)
    - Server sends: JSON {"text": "...", "is_final": false}
    """
    await ws.accept()
    session_id = str(uuid.uuid4())

    # Create streaming session
    session = StreamingSession(model, session_id)
    streaming_sessions[session_id] = session

    logger.info(f"New streaming session: {session_id}")

    try:
        while True:
            # Receive audio chunk (expect PCM int16 bytes)
            data = await ws.receive_bytes()

            # Convert to float32 normalized audio
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # Process through streaming buffer
            transcript = session.process_audio(audio_float32)

            # Send back transcript if any
            if transcript:
                await ws.send_json({
                    "text": transcript,
                    "is_final": False,
                    "session_id": session_id
                })

    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"Error in session {session_id}: {e}")
        await ws.close(code=1011, reason=str(e))
    finally:
        # Cleanup
        if session_id in streaming_sessions:
            del streaming_sessions[session_id]
        logger.info(f"Session {session_id} cleaned up")


def setup_model_for_streaming(model):
    """
    Configure model for low-latency streaming inference.
    Enables cache-aware attention if available.
    """
    try:
        # Enable cache-aware attention for streaming
        model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=[256, 64]  # [left, right] context chunks
        )
        logger.info("Model configured for streaming with cache-aware attention")
    except Exception as e:
        logger.warning(f"Could not enable cache-aware attention: {e}")
        logger.info("Model will use standard inference")

    return model
