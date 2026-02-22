"""
Buffered streaming STT/translation server for Canary 1B V2.

Canary is an encoder-decoder (FastConformer + Transformer) model, so it cannot
do frame-level cache-aware streaming like CTC/TDT models.  Instead we accumulate
audio in a growing buffer and periodically run full model.transcribe() to produce
interim results, then emit a final result on silence or explicit end_utterance.
"""

import asyncio
import json
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .config import (
    SILENCE_TIMEOUT_S,
    SUPPORTED_LANGUAGES,
    TARGET_SR,
    TRANSCRIBE_INTERVAL_S,
    logger,
)


# ---------------------------------------------------------------------------
# Streaming session
# ---------------------------------------------------------------------------


class StreamingSession:
    """One WebSocket connection = one session with fixed source/target language."""

    def __init__(
        self,
        model,
        batcher,
        session_id: str,
        source_lang: str = "en",
        target_lang: str = "en",
    ):
        self.model = model
        self.batcher = batcher
        self.session_id = session_id
        self.source_lang = source_lang
        self.target_lang = target_lang

        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_audio_time = time.time()
        self.last_transcribe_time = 0.0
        self.last_transcript = ""
        self.is_active = True

    # -- audio accumulation --------------------------------------------------

    def add_audio(self, audio_bytes: bytes):
        """Append PCM int16 audio bytes to the session buffer."""
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_float32])
        self.last_audio_time = time.time()

    # -- scheduling helpers --------------------------------------------------

    def should_transcribe(self) -> bool:
        has_audio = len(self.audio_buffer) > int(TARGET_SR * 0.5)  # >= 500 ms
        interval_passed = (time.time() - self.last_transcribe_time) >= TRANSCRIBE_INTERVAL_S
        return has_audio and interval_passed

    def is_silent(self) -> bool:
        return (time.time() - self.last_audio_time) >= SILENCE_TIMEOUT_S

    # -- transcription -------------------------------------------------------

    async def transcribe_async(self) -> str:
        """Transcribe via the batcher (batched GPU inference)."""
        text = await self.batcher.transcribe_buffer(
            self.audio_buffer.copy(),
            self.source_lang,
            self.target_lang,
        )
        self.last_transcribe_time = time.time()
        return text

    # -- reset ---------------------------------------------------------------

    def reset(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_transcript = ""
        self.last_transcribe_time = 0.0


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------


async def websocket_streaming_endpoint(ws: WebSocket, model, batcher):
    """
    WebSocket streaming endpoint for Canary 1B V2.

    Protocol
    --------
    1. Client connects to ``/stream``
    2. Client sends JSON config::

           {"source_lang": "en", "target_lang": "en"}

    3. Server replies::

           {"status": "ready", "session_id": "...", ...}

    4. Client streams PCM int16 audio (16 kHz mono, 80 ms chunks recommended)
    5. Server sends interim / final JSON transcripts::

           {"text": "...", "is_final": false, "session_id": "..."}

    6. Client may send commands at any time::

           {"action": "end_utterance"}   — force final result
           {"action": "reset"}           — clear buffer
    """
    await ws.accept()
    session: Optional[StreamingSession] = None

    try:
        # ---- 1. language config handshake --------------------------------
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
            config = json.loads(raw)
        except asyncio.TimeoutError:
            await ws.close(code=1008, reason="Config timeout — send language JSON first")
            return
        except (json.JSONDecodeError, KeyError):
            await ws.close(code=1008, reason="Invalid config JSON")
            return

        source_lang = config.get("source_lang", "en")
        target_lang = config.get("target_lang", "en")

        if source_lang not in SUPPORTED_LANGUAGES:
            await ws.send_json({
                "error": f"Unsupported source_lang: {source_lang}",
                "supported": SUPPORTED_LANGUAGES,
            })
            await ws.close()
            return
        if target_lang not in SUPPORTED_LANGUAGES:
            await ws.send_json({
                "error": f"Unsupported target_lang: {target_lang}",
                "supported": SUPPORTED_LANGUAGES,
            })
            await ws.close()
            return

        session_id = str(uuid.uuid4())
        session = StreamingSession(model, batcher, session_id, source_lang, target_lang)

        await ws.send_json({
            "status": "ready",
            "session_id": session_id,
            "source_lang": source_lang,
            "target_lang": target_lang,
        })
        logger.info(
            "Session %s: started (src=%s, tgt=%s)",
            session_id, source_lang, target_lang,
        )

        # ---- 2. silence monitor ------------------------------------------
        async def silence_monitor():
            while session.is_active:
                await asyncio.sleep(0.5)
                if session.is_silent() and len(session.audio_buffer) > 0:
                    text = await session.transcribe_async()
                    if text:
                        try:
                            await ws.send_json({
                                "text": text,
                                "is_final": True,
                                "session_id": session_id,
                            })
                        except Exception:
                            break
                    session.reset()

        silence_task = asyncio.create_task(silence_monitor())

        # ---- 3. main receive loop ----------------------------------------
        try:
            while True:
                message = await ws.receive()

                if "bytes" in message:
                    session.add_audio(message["bytes"])

                    if session.should_transcribe():
                        text = await session.transcribe_async()
                        if text and text != session.last_transcript:
                            session.last_transcript = text
                            await ws.send_json({
                                "text": text,
                                "is_final": False,
                                "session_id": session_id,
                            })

                elif "text" in message:
                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue

                    action = data.get("action")

                    if action == "end_utterance":
                        text = await session.transcribe_async()
                        if text:
                            await ws.send_json({
                                "text": text,
                                "is_final": True,
                                "session_id": session_id,
                            })
                        session.reset()

                    elif action == "reset":
                        session.reset()
                        await ws.send_json({
                            "status": "reset",
                            "session_id": session_id,
                        })
        finally:
            session.is_active = False
            silence_task.cancel()
            try:
                await silence_task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logger.info(
            "Session %s: disconnected",
            session.session_id if session else "?",
        )
    except Exception as e:
        logger.error("Session error: %s", e, exc_info=True)
        try:
            await ws.close(code=1011, reason=str(e)[:120])
        except Exception:
            pass
