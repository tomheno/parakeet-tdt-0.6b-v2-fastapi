"""
LiveKit STT plugin for Parakeet TDT streaming inference.
Connects to Parakeet WebSocket server for speech-to-text.

Usage:
    from livekit_plugin import ParakeetSTT

    stt = ParakeetSTT(url="ws://localhost:8000/stream")
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
from dataclasses import dataclass
from typing import Optional
import logging
import json

# LiveKit imports
try:
    from livekit.agents import stt, utils
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    logging.warning("LiveKit not installed. Plugin will not work.")

# WebSocket client
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets not installed. Install with: uv pip install websockets")

logger = logging.getLogger(__name__)


@dataclass
class ParakeetOptions:
    """Configuration for Parakeet STT."""
    url: str = "ws://localhost:8000/stream"  # WebSocket server URL
    language: str = "en"
    sample_rate: int = 16000
    silence_timeout: float = 1.5  # Seconds of silence before end_utterance


if LIVEKIT_AVAILABLE and WEBSOCKETS_AVAILABLE:
    class ParakeetSTT(stt.STT):
        """
        Parakeet TDT streaming STT for LiveKit agents.

        Connects to Parakeet WebSocket server for low-latency (<300ms) speech-to-text.
        """

        def __init__(
            self,
            *,
            url: str = "ws://localhost:8000/stream",
            language: str = "en",
            sample_rate: int = 16000,
            silence_timeout: float = 1.5
        ):
            super().__init__(
                capabilities=stt.STTCapabilities(
                    streaming=True,
                    interim_results=True
                )
            )

            self._opts = ParakeetOptions(
                url=url,
                language=language,
                sample_rate=sample_rate,
                silence_timeout=silence_timeout
            )

            logger.info(f"Parakeet STT initialized with URL: {url}")

        def stream(
            self,
            *,
            language: Optional[str] = None
        ) -> ParakeetStream:
            """Create a new streaming session."""
            return ParakeetStream(
                stt=self,
                opts=self._opts,
                language=language or self._opts.language
            )


    class ParakeetStream(stt.SpeechStream):
        """Streaming session for Parakeet STT."""

        def __init__(
            self,
            *,
            stt: ParakeetSTT,
            opts: ParakeetOptions,
            language: str
        ):
            super().__init__(stt=stt, sample_rate=opts.sample_rate)

            self._opts = opts
            self._language = language
            self._websocket: Optional[websockets.WebSocketClientProtocol] = None
            self._ws_task: Optional[asyncio.Task] = None
            self._speaking = False
            self._last_transcript = ""

        async def _run(self):
            """Main processing loop."""
            try:
                # Connect to WebSocket server
                logger.info(f"Connecting to Parakeet server: {self._opts.url}")
                self._websocket = await websockets.connect(
                    self._opts.url,
                    ping_interval=None,  # Disable ping/pong
                    close_timeout=5
                )
                logger.info("Connected to Parakeet server")

                # Start WebSocket receiver task
                self._ws_task = asyncio.create_task(self._receive_loop())

                # Process audio frames
                async for frame in self._input_ch:
                    if isinstance(frame, rtc.AudioFrame):
                        await self._process_frame(frame)
                    elif isinstance(frame, self._FlushSentinel):
                        # End of speech - request final transcript
                        if self._speaking:
                            await self._end_utterance()

            except Exception as e:
                logger.error(f"Error in streaming: {e}", exc_info=True)
            finally:
                # Cleanup
                await self._cleanup()

        async def _process_frame(self, frame: rtc.AudioFrame):
            """Process a single audio frame and send to WebSocket."""
            if not self._websocket:
                return

            try:
                # Convert to numpy array (int16 PCM)
                audio_data = np.frombuffer(
                    frame.data.tobytes(),
                    dtype=np.int16
                )

                # Send to WebSocket server
                await self._websocket.send(audio_data.tobytes())

            except Exception as e:
                logger.error(f"Error processing frame: {e}")

        async def _receive_loop(self):
            """Receive transcripts from WebSocket server."""
            if not self._websocket:
                return

            try:
                async for message in self._websocket:
                    if isinstance(message, str):
                        # JSON message from server
                        try:
                            data = json.loads(message)
                            text = data.get("text", "")
                            is_final = data.get("is_final", False)

                            if text:
                                # Start of speech event
                                if not self._speaking:
                                    self._speaking = True
                                    self._event_ch.send_nowait(
                                        stt.SpeechEvent(
                                            type=stt.SpeechEventType.START_OF_SPEECH
                                        )
                                    )

                                # Interim or final transcript
                                if is_final:
                                    # Final transcript
                                    self._event_ch.send_nowait(
                                        stt.SpeechEvent(
                                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                            alternatives=[
                                                stt.SpeechData(
                                                    text=text,
                                                    language=self._language
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

                                else:
                                    # Interim transcript
                                    self._event_ch.send_nowait(
                                        stt.SpeechEvent(
                                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                            alternatives=[
                                                stt.SpeechData(
                                                    text=text,
                                                    language=self._language
                                                )
                                            ]
                                        )
                                    )
                                    self._last_transcript = text

                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from server: {message}")

            except Exception as e:
                logger.error(f"Error receiving from WebSocket: {e}")

        async def _end_utterance(self):
            """Request final transcript from server."""
            if not self._websocket:
                return

            try:
                # Send end_utterance command
                await self._websocket.send(json.dumps({"action": "end_utterance"}))
            except Exception as e:
                logger.error(f"Error sending end_utterance: {e}")

        async def _cleanup(self):
            """Cleanup WebSocket connection."""
            # Cancel receiver task
            if self._ws_task:
                self._ws_task.cancel()
                try:
                    await self._ws_task
                except asyncio.CancelledError:
                    pass

            # Close WebSocket
            if self._websocket:
                try:
                    await self._websocket.close()
                except:
                    pass

            logger.info("Parakeet stream cleaned up")

else:
    # Stub classes if dependencies not available
    class ParakeetSTT:
        def __init__(self, **kwargs):
            missing = []
            if not LIVEKIT_AVAILABLE:
                missing.append("livekit-agents")
            if not WEBSOCKETS_AVAILABLE:
                missing.append("websockets")

            raise ImportError(
                f"Required packages not installed: {', '.join(missing)}. "
                f"Install with: uv sync --extra livekit"
            )

    class ParakeetStream:
        pass
