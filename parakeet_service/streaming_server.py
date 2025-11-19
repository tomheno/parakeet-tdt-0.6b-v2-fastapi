"""
True streaming STT server using NeMo's cache-aware streaming inference.
Low-latency alternative to batch processing.
"""
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
import torch
import asyncio
import uuid
import time
from typing import Dict, Optional
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

    def __init__(self, model, session_id: str, silence_timeout: float = 1.5):
        self.session_id = session_id
        self.model = model
        self.silence_timeout = silence_timeout  # Seconds of silence to trigger final

        # State tracking
        self.last_transcript = ""
        self.last_transcript_time = 0.0
        self.is_speaking = False

        if STREAMING_AVAILABLE:
            # Use NeMo's cache-aware streaming buffer
            # Note: chunk_size, shift_size etc. are configured via model.encoder.streaming_cfg
            # online_normalization=False because model's preprocessor already normalizes
            # (and NeMo has a device mismatch bug when online_normalization=True)
            self.buffer = CacheAwareStreamingAudioBuffer(
                model=model,
                online_normalization=False
            )
            logger.info(f"Session {session_id}: Using CacheAwareStreamingAudioBuffer (offline normalization)")
        else:
            # Fallback: simple chunked inference
            self.buffer = None
            self.audio_buffer = []
            logger.info(f"Session {session_id}: Using fallback chunked inference")

    def process_audio(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Process audio chunk and return transcript (if any)."""
        if STREAMING_AVAILABLE and self.buffer is not None:
            try:
                # Append audio to streaming buffer and process
                self.buffer.append_audio(audio_chunk)

                # Process chunks from buffer
                transcripts = []
                for audio_signal, audio_signal_len in self.buffer:
                    # Transcribe using model's streaming inference
                    with torch.inference_mode():
                        logits = self.model.transcribe(
                            audio=audio_signal,
                            logprobs=False
                        )
                        if logits and len(logits) > 0:
                            transcripts.append(logits[0])

                # Reset buffer pointer for next chunk
                if not self.buffer.is_buffer_empty():
                    self.buffer.reset_buffer_pointer()

                # Join transcripts if any
                if transcripts:
                    result = " ".join(transcripts).strip()
                    if result:
                        self.last_transcript = result
                        self.last_transcript_time = time.time()
                        if not self.is_speaking:
                            self.is_speaking = True
                        return self.last_transcript
                return None
            except Exception as e:
                logger.error(f"Streaming inference error: {e}", exc_info=True)
                return None
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
                        if result and result[0]:
                            self.last_transcript = result[0]
                            self.last_transcript_time = time.time()
                            if not self.is_speaking:
                                self.is_speaking = True
                            return self.last_transcript
                    except Exception as e:
                        logger.error(f"Fallback inference error: {e}")
            return None

    def check_silence(self) -> bool:
        """Check if silence timeout has been reached."""
        if not self.is_speaking:
            return False

        if self.last_transcript and self.last_transcript_time > 0:
            silence_duration = time.time() - self.last_transcript_time
            return silence_duration >= self.silence_timeout
        return False

    def get_final_transcript(self) -> Optional[str]:
        """Get final transcript and reset state."""
        if self.last_transcript:
            final = self.last_transcript
            self.reset()
            return final
        return None

    def reset(self):
        """Reset session state."""
        if STREAMING_AVAILABLE and self.buffer is not None:
            self.buffer.reset_buffer()
        else:
            self.audio_buffer = []

        self.last_transcript = ""
        self.last_transcript_time = 0.0
        self.is_speaking = False


# Global session storage
streaming_sessions: Dict[str, StreamingSession] = {}


async def websocket_streaming_endpoint(ws: WebSocket, model):
    """
    WebSocket endpoint for real-time streaming STT.

    Protocol:
    - Client sends:
      * PCM int16 audio bytes (16kHz mono) for transcription
      * JSON {"action": "end_utterance"} to force final transcript
      * JSON {"action": "reset"} to reset session
    - Server sends:
      * JSON {"text": "...", "is_final": false, "session_id": "..."} for interim
      * JSON {"text": "...", "is_final": true, "session_id": "..."} for final
    """
    await ws.accept()
    session_id = str(uuid.uuid4())

    # Create streaming session
    session = StreamingSession(model, session_id, silence_timeout=1.5)
    streaming_sessions[session_id] = session

    logger.info(f"New streaming session: {session_id}")

    # Task for silence detection
    silence_check_task = None

    async def check_silence_loop():
        """Periodically check for silence and send final transcript."""
        while True:
            await asyncio.sleep(0.5)  # Check every 500ms

            if session.check_silence():
                final_transcript = session.get_final_transcript()
                if final_transcript:
                    try:
                        await ws.send_json({
                            "text": final_transcript,
                            "is_final": True,
                            "session_id": session_id
                        })
                        logger.info(f"Session {session_id}: Sent final transcript")
                    except Exception as e:
                        logger.error(f"Error sending final transcript: {e}")
                        break

    try:
        # Start silence detection task
        silence_check_task = asyncio.create_task(check_silence_loop())

        while True:
            # Try to receive data (bytes or text)
            try:
                # Check if message is binary (audio) or text (command)
                message = await asyncio.wait_for(ws.receive(), timeout=0.1)

                if "bytes" in message:
                    # Audio data
                    data = message["bytes"]

                    # Convert to float32 normalized audio
                    audio_int16 = np.frombuffer(data, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0

                    # Process through streaming buffer
                    transcript = session.process_audio(audio_float32)

                    # Send back interim transcript if any
                    if transcript:
                        await ws.send_json({
                            "text": transcript,
                            "is_final": False,
                            "session_id": session_id
                        })

                elif "text" in message:
                    # Command message
                    import json
                    try:
                        command = json.loads(message["text"])
                        action = command.get("action")

                        if action == "end_utterance":
                            # Force final transcript
                            final_transcript = session.get_final_transcript()
                            if final_transcript:
                                await ws.send_json({
                                    "text": final_transcript,
                                    "is_final": True,
                                    "session_id": session_id
                                })

                        elif action == "reset":
                            # Reset session
                            session.reset()
                            await ws.send_json({
                                "status": "reset",
                                "session_id": session_id
                            })

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON command: {message['text']}")

            except asyncio.TimeoutError:
                # No message received, continue
                continue

    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"Error in session {session_id}: {e}", exc_info=True)
        try:
            await ws.close(code=1011, reason=str(e))
        except:
            pass
    finally:
        # Cancel silence check task
        if silence_check_task:
            silence_check_task.cancel()
            try:
                await silence_check_task
            except asyncio.CancelledError:
                pass

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
