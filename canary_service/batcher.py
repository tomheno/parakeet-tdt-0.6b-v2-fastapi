"""
Dynamic request batcher for Canary 1B V2.

Instead of processing requests one-at-a-time behind a lock, we collect
pending requests into batches and run model.transcribe() once per batch.
This keeps the GPU saturated and dramatically improves throughput under
concurrency.
"""

import asyncio
import os
import tempfile
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from .config import TARGET_SR, logger

# ---------------------------------------------------------------------------
# Batch config
# ---------------------------------------------------------------------------

MAX_BATCH_SIZE = 64        # Max files per model.transcribe() call
BATCH_WAIT_MS = 20         # Max ms to wait for more requests before dispatching
MAX_BATCH_WAIT_MS = 100    # Absolute max wait time


@dataclass
class TranscribeRequest:
    """A single pending transcription request."""
    audio_path: str
    source_lang: str
    target_lang: str
    timestamps: bool = False
    future: asyncio.Future = field(default_factory=lambda: None)
    submitted_at: float = field(default_factory=time.monotonic)


class InferenceBatcher:
    """Collects transcription requests and dispatches them in batches.

    Usage::

        batcher = InferenceBatcher(model)
        batcher.start()

        # From async code:
        result = await batcher.transcribe(path, src, tgt)

        batcher.stop()
    """

    def __init__(self, model, max_batch_size: int = MAX_BATCH_SIZE):
        self.model = model
        self.max_batch_size = max_batch_size
        self._queue: asyncio.Queue[TranscribeRequest] = None
        self._loop: asyncio.AbstractEventLoop = None
        self._task: asyncio.Task = None
        self._started = False

    def start(self, loop: asyncio.AbstractEventLoop = None):
        self._loop = loop or asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._task = self._loop.create_task(self._batch_loop())
        self._started = True
        logger.info("InferenceBatcher started (max_batch=%d)", self.max_batch_size)

    def stop(self):
        if self._task:
            self._task.cancel()
        self._started = False

    async def transcribe(
        self,
        audio_path: str,
        source_lang: str,
        target_lang: str,
        timestamps: bool = False,
    ):
        """Submit a request and wait for the result."""
        future = self._loop.create_future()
        req = TranscribeRequest(
            audio_path=audio_path,
            source_lang=source_lang,
            target_lang=target_lang,
            timestamps=timestamps,
            future=future,
        )
        await self._queue.put(req)
        return await future

    async def transcribe_buffer(
        self,
        audio_buffer: np.ndarray,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Write buffer to temp file, submit, and return text."""
        if len(audio_buffer) < int(TARGET_SR * 0.1):
            return ""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                sf.write(tmp_path, audio_buffer, TARGET_SR)
            result = await self.transcribe(tmp_path, source_lang, target_lang)
            text = result[0].text if hasattr(result[0], "text") else str(result[0])
            return text.strip()
        except Exception as e:
            logger.error("Transcribe buffer error: %s", e)
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _batch_loop(self):
        """Main loop: collect requests, dispatch batches."""
        logger.info("Batch loop running")
        while True:
            try:
                # Wait for at least one request
                first = await self._queue.get()
                batch = [first]

                # Collect more requests up to max_batch_size with a short wait
                deadline = time.monotonic() + BATCH_WAIT_MS / 1000
                abs_deadline = first.submitted_at + MAX_BATCH_WAIT_MS / 1000

                while len(batch) < self.max_batch_size:
                    now = time.monotonic()
                    wait = min(deadline - now, abs_deadline - now)
                    if wait <= 0:
                        break
                    try:
                        req = await asyncio.wait_for(self._queue.get(), timeout=wait)
                        batch.append(req)
                    except asyncio.TimeoutError:
                        break

                # Group by (source_lang, target_lang, timestamps) for batching
                groups = {}
                for req in batch:
                    key = (req.source_lang, req.target_lang, req.timestamps)
                    groups.setdefault(key, []).append(req)

                # Dispatch each group
                for (src, tgt, ts), reqs in groups.items():
                    paths = [r.audio_path for r in reqs]
                    try:
                        outputs = await self._run_batch(paths, src, tgt, ts)
                        for req, out in zip(reqs, outputs):
                            if not req.future.done():
                                req.future.set_result([out])
                    except Exception as e:
                        for req in reqs:
                            if not req.future.done():
                                req.future.set_exception(e)

                if len(batch) > 1:
                    logger.debug("Dispatched batch of %d requests", len(batch))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Batch loop error: %s", e, exc_info=True)

    async def _run_batch(self, paths, source_lang, target_lang, timestamps):
        """Run model.transcribe on a batch in a thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._run_batch_sync,
            paths, source_lang, target_lang, timestamps,
        )

    def _run_batch_sync(self, paths, source_lang, target_lang, timestamps):
        """Blocking batch transcription."""
        with torch.inference_mode():
            outputs = self.model.transcribe(
                paths,
                source_lang=source_lang,
                target_lang=target_lang,
                timestamps=timestamps,
                batch_size=len(paths),
            )
        return outputs
