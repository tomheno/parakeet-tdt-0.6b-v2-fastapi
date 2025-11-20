"""
Low-latency streaming STT server with VAD-based utterance detection.
Combines NeMo cache-aware streaming + Silero VAD for optimal latency and accuracy.
"""
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
import torch
import asyncio
import uuid
import time
from typing import Dict, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# NeMo streaming imports
try:
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    STREAMING_AVAILABLE = True
except ImportError:
    logger.warning("NeMo streaming utils not available")
    STREAMING_AVAILABLE = False

# Silero VAD imports
try:
    from torch.hub import load as torch_hub_load
    vad_model, vad_utils = torch_hub_load("snakers4/silero-vad", "silero_vad", force_reload=False)
    (_, _, _, VADIterator, _) = vad_utils
    VAD_AVAILABLE = True
    logger.info("Silero VAD loaded successfully")
except Exception as e:
    logger.warning(f"Silero VAD not available: {e}")
    VAD_AVAILABLE = False


class SpeechState(Enum):
    """Speech detection states"""
    SILENCE = "silence"
    SPEECH = "speech"
    TRAILING = "trailing"  # Speech ended, waiting to finalize


class StreamingSessionVAD:
    """
    Manages a single streaming STT session with VAD-based utterance detection.

    Low-latency design:
    - VAD processes 32ms windows (512 samples @ 16kHz)
    - Streaming inference runs on speech frames immediately
    - No batching delay for interim transcripts
    """

    def __init__(
        self,
        model,
        session_id: str,
        sample_rate: int = 16000,
        vad_threshold: float = 0.6,
        min_silence_ms: int = 250,
        speech_pad_ms: int = 120,
    ):
        self.session_id = session_id
        self.model = model
        self.sample_rate = sample_rate

        # VAD configuration
        self.vad_threshold = vad_threshold
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms

        # State tracking
        self.state = SpeechState.SILENCE
        self.accumulated_transcript = ""
        self.last_emission_time = 0.0

        # Initialize VAD if available
        if VAD_AVAILABLE:
            self.vad = VADIterator(
                vad_model,
                sampling_rate=sample_rate,
                threshold=vad_threshold,
                min_silence_duration_ms=min_silence_ms,
                speech_pad_ms=speech_pad_ms,
            )
            logger.info(f"Session {session_id}: VAD enabled (threshold={vad_threshold})")
        else:
            self.vad = None
            logger.warning(f"Session {session_id}: VAD not available, using fallback")

        # Initialize streaming buffer if available
        if STREAMING_AVAILABLE:
            self.buffer = CacheAwareStreamingAudioBuffer(
                model=model,
                online_normalization=False  # Preprocessor handles normalization
            )
            logger.info(f"Session {session_id}: Streaming buffer enabled")
        else:
            self.buffer = None
            self.audio_buffer = []
            logger.info(f"Session {session_id}: Using fallback chunked inference")

    def process_audio(self, audio_chunk: np.ndarray) -> tuple[Optional[str], bool]:
        """
        Process audio chunk and return (interim_transcript, is_final).

        Returns:
            (str, False) - Interim transcript during speech
            (str, True)  - Final transcript when utterance ends
            (None, False) - No transcript yet
        """
        # Run VAD if available
        is_speech = True
        utterance_ended = False

        if self.vad is not None:
            # Process through VAD (expects 512-sample windows)
            vad_event = self._run_vad(audio_chunk)

            if vad_event:
                if vad_event.get("start"):
                    self.state = SpeechState.SPEECH
                    logger.debug(f"Session {self.session_id}: Speech started")

                elif vad_event.get("end"):
                    self.state = SpeechState.TRAILING
                    utterance_ended = True
                    logger.debug(f"Session {self.session_id}: Speech ended")

            is_speech = self.state == SpeechState.SPEECH

        # Process through streaming inference if speech detected
        if is_speech or utterance_ended:
            transcript = self._run_streaming_inference(audio_chunk)

            if transcript:
                self.accumulated_transcript = transcript
                self.last_emission_time = time.time()

            # Return final transcript if utterance ended
            if utterance_ended:
                final = self.accumulated_transcript
                self.reset()
                return (final, True) if final else (None, False)

            # Return interim transcript
            return (self.accumulated_transcript, False) if self.accumulated_transcript else (None, False)

        return (None, False)

    def _run_vad(self, audio_chunk: np.ndarray):
        """Run VAD on audio chunk (processes in 512-sample windows)."""
        try:
            # VAD expects 512 samples (32ms @ 16kHz)
            window_size = 512
            events = []

            for start in range(0, len(audio_chunk), window_size):
                window = audio_chunk[start:start + window_size]
                if len(window) < window_size:
                    break  # Wait for full window

                event = self.vad(window, return_seconds=False)
                if event:
                    events.append(event)

            # Return most recent event
            return events[-1] if events else None

        except Exception as e:
            logger.error(f"VAD error: {e}")
            return None

    def _run_streaming_inference(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Run streaming inference on audio chunk."""
        if STREAMING_AVAILABLE and self.buffer is not None:
            try:
                # Append audio to streaming buffer
                self.buffer.append_audio(audio_chunk)

                # Process chunks from buffer
                transcripts = []
                for audio_signal, audio_signal_len in self.buffer:
                    with torch.inference_mode():
                        # Encode features (audio_signal is preprocessed mel-spec)
                        encoded, encoded_len = self.model(
                            processed_signal=audio_signal,
                            processed_signal_length=audio_signal_len
                        )

                        # Decode with greedy decoding
                        best_hyp, _ = self.model.decoding.rnnt_decoder_predictions_tensor(
                            encoder_output=encoded,
                            encoded_lengths=encoded_len,
                            return_hypotheses=False
                        )

                        if best_hyp and len(best_hyp) > 0:
                            transcripts.append(best_hyp[0])

                # Reset buffer pointer for next chunk
                if not self.buffer.is_buffer_empty():
                    self.buffer.reset_buffer_pointer()

                # Join transcripts
                if transcripts:
                    return " ".join(transcripts).strip()

            except Exception as e:
                logger.error(f"Streaming inference error: {e}", exc_info=True)

        else:
            # Fallback: accumulate and transcribe
            self.audio_buffer.append(audio_chunk)
            if len(self.audio_buffer) >= 10:  # ~640ms
                audio = np.concatenate(self.audio_buffer)
                self.audio_buffer = []

                import tempfile
                import soundfile as sf

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, audio, self.sample_rate)
                    try:
                        with torch.inference_mode():
                            result = self.model.transcribe([tmp.name], batch_size=1)
                        if result and result[0]:
                            return result[0]
                    except Exception as e:
                        logger.error(f"Fallback inference error: {e}")

        return None

    def reset(self):
        """Reset session state."""
        if self.vad is not None:
            self.vad.reset_states()

        if STREAMING_AVAILABLE and self.buffer is not None:
            self.buffer.reset_buffer()
        else:
            self.audio_buffer = []

        self.state = SpeechState.SILENCE
        self.accumulated_transcript = ""
        self.last_emission_time = 0.0


# Global session storage
streaming_sessions: Dict[str, StreamingSessionVAD] = {}


async def websocket_streaming_endpoint_vad(ws: WebSocket, model):
    """
    WebSocket endpoint for low-latency streaming STT with VAD.

    Protocol:
    - Client sends: PCM int16 audio bytes (16kHz mono)
    - Server sends:
      * {"text": "...", "is_final": false} - Interim transcript
      * {"text": "...", "is_final": true}  - Final transcript (utterance ended)
    """
    await ws.accept()
    session_id = str(uuid.uuid4())

    # Create streaming session with VAD
    session = StreamingSessionVAD(
        model,
        session_id,
        sample_rate=16000,
        vad_threshold=0.6,      # Voice probability threshold
        min_silence_ms=250,     # Finalize after 250ms silence
        speech_pad_ms=120,      # Keep 120ms context
    )
    streaming_sessions[session_id] = session

    logger.info(f"New streaming session with VAD: {session_id}")

    try:
        while True:
            try:
                # Receive audio data
                message = await asyncio.wait_for(ws.receive(), timeout=0.1)

                if "bytes" in message:
                    # Convert to float32 normalized audio
                    audio_int16 = np.frombuffer(message["bytes"], dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0

                    # Process audio (returns interim or final transcript)
                    transcript, is_final = session.process_audio(audio_float32)

                    # Send transcript if any
                    if transcript:
                        await ws.send_json({
                            "text": transcript,
                            "is_final": is_final,
                            "session_id": session_id
                        })

                        if is_final:
                            logger.info(f"Session {session_id}: Final transcript sent")

                elif "text" in message:
                    # Command message (reset, etc.)
                    import json
                    try:
                        command = json.loads(message["text"])
                        if command.get("action") == "reset":
                            session.reset()
                            await ws.send_json({
                                "status": "reset",
                                "session_id": session_id
                            })
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON: {message['text']}")

            except asyncio.TimeoutError:
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
        # Cleanup
        if session_id in streaming_sessions:
            del streaming_sessions[session_id]
        logger.info(f"Session {session_id} cleaned up")


def setup_model_for_streaming(model):
    """
    Configure model for low-latency streaming inference.
    """
    try:
        model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=[256, 64]  # [left, right] context
        )
        logger.info("Model configured for streaming with cache-aware attention")
    except Exception as e:
        logger.warning(f"Could not enable cache-aware attention: {e}")

    return model
