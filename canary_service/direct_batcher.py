"""
In-flight (continuous) batcher for Canary 1B V2.

Inspired by vLLM's continuous batching: no artificial wait — the GPU always
processes the maximum available batch.  Requests join immediately as they
arrive; completed requests return without waiting for the whole batch.

Pipeline (pipelined):
  1. Audio arrives as bytes → decoded to numpy on CPU (parallel with GPU)
  2. Queued as ready numpy arrays
  3. Batch loop drains queue instantly (no wait if items available)
  4. GPU inference runs on dedicated thread (pinned, no contention)
  5. While GPU runs, next batch is collected from queue
  6. Results returned to individual callers immediately

Optimizations:
  - Dedicated single-thread GPU executor (no thread pool contention)
  - Pre-allocated GPU tensor buffer (reused across batches, avoids alloc)
  - Pipelined: next batch collected while GPU processes current batch
  - .expand() instead of .repeat() for prompt tokens (zero-copy)
  - CUDA non-blocking transfers for CPU→GPU data movement
  - Larger audio decode pool (32 workers)
  - torch.compile(mode="reduce-overhead") on encoder (uses CUDA graphs)
"""

import asyncio
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch

from .config import TARGET_SR, logger

# Lazy imports — only needed when timestamps=True
_ts_funcs = {}

def _get_timestamp_funcs():
    if not _ts_funcs:
        from nemo.collections.asr.parts.utils.timestamp_utils import (
            get_forced_aligned_timestamps_with_external_model,
            process_aed_timestamp_outputs,
        )
        _ts_funcs["ctc_align"] = get_forced_aligned_timestamps_with_external_model
        _ts_funcs["aed_parse"] = process_aed_timestamp_outputs
    return _ts_funcs

# ---------------------------------------------------------------------------
# torch.compile config
# ---------------------------------------------------------------------------

# Compilation strategy:
#   "reduce-overhead" — CUDA graphs, fastest steady-state but needs per-shape warmup
#   "max-autotune"    — picks best kernel (Triton vs CUDA graphs) per op
#   "default"         — balanced compile, works well with dynamic=True
#
# dynamic=True uses symbolic shapes so ONE compilation covers all batch sizes.
COMPILE_ENCODER = True
COMPILE_MODE = "reduce-overhead"
COMPILE_DYNAMIC = False  # NeMo conformer has ops incompatible with symbolic shapes

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_BATCH_SIZE = 256

# Thread pool for CPU-bound audio decoding (runs in parallel with GPU)
_audio_pool = ThreadPoolExecutor(max_workers=32)

# Dedicated single-thread executor for GPU work (avoids GIL contention)
_gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TranscribeRequest:
    """A pending transcription request with pre-decoded audio."""
    audio: np.ndarray           # float32 mono PCM at TARGET_SR
    source_lang: str
    target_lang: str
    timestamps: bool = False
    future: asyncio.Future = field(default_factory=lambda: None)


@dataclass
class SimpleHypothesis:
    """Lightweight result container."""
    text: str
    timestamp: Optional[dict] = None


# ---------------------------------------------------------------------------
# Audio decoding (runs on CPU thread pool, parallel with GPU)
# ---------------------------------------------------------------------------

def _decode_audio(audio_data) -> np.ndarray:
    """Decode audio from bytes, numpy, or file path → float32 mono numpy."""
    if isinstance(audio_data, np.ndarray):
        return audio_data.astype(np.float32) if audio_data.dtype != np.float32 else audio_data
    if isinstance(audio_data, (bytes, bytearray)):
        data, sr = sf.read(io.BytesIO(audio_data), dtype="float32")
    elif isinstance(audio_data, str):
        data, sr = sf.read(audio_data, dtype="float32")
    else:
        raise ValueError(f"Unsupported audio type: {type(audio_data)}")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        # Simple linear interpolation resampling (fast, good enough for ASR)
        indices = np.linspace(0, len(data) - 1, int(len(data) * TARGET_SR / sr))
        data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
    return data


# ---------------------------------------------------------------------------
# In-flight batcher
# ---------------------------------------------------------------------------

class DirectInferenceBatcher:
    """Continuous in-flight batcher: GPU always processes max available batch.

    No artificial wait — if 1 item is queued, process 1. If 64 are queued,
    process 64.  At high load, batches fill naturally from the queue.

    Pipelined: while GPU processes batch N, the event loop collects batch N+1.
    """

    def __init__(self, model, max_batch_size: int = MAX_BATCH_SIZE):
        self.model = model
        self.max_batch_size = max_batch_size
        self._queue: asyncio.Queue[TranscribeRequest] = None
        self._loop: asyncio.AbstractEventLoop = None
        self._task: asyncio.Task = None
        self._started = False

        self._prompt_cache: dict[tuple, torch.Tensor] = {}
        self._device = next(model.parameters()).device
        self._dtype = next(model.parameters()).dtype

        # Timestamp processing: prefer CTC forced alignment (accurate),
        # fall back to decoder token parsing if CTC model not available
        self._timestamps_asr_model = getattr(model, 'timestamps_asr_model', None)
        self._subsampling_factor = model.encoder.subsampling_factor
        self._window_stride = model.cfg['preprocessor']['window_stride']
        if self._timestamps_asr_model is not None:
            logger.info("Timestamps: CTC forced alignment model available")
        else:
            logger.info("Timestamps: using decoder token parsing (no CTC model)")

        # Pre-allocated GPU buffer for audio (reused across batches)
        # Sized for max_batch_size × 30s audio at 16kHz
        self._max_audio_samples = TARGET_SR * 30  # 30s max
        self._audio_buf = torch.zeros(
            max_batch_size, self._max_audio_samples,
            dtype=torch.float32, device=self._device,
        )
        self._length_buf = torch.zeros(
            max_batch_size, dtype=torch.long, device=self._device,
        )

        # Pin a CPU staging buffer for fast async transfer
        self._cpu_staging = torch.zeros(
            max_batch_size, self._max_audio_samples,
            dtype=torch.float32, pin_memory=True,
        )

        # torch.compile the encoder only (preprocessor is ~1ms, not worth compiling)
        if COMPILE_ENCODER:
            try:
                self._compiled_encoder = torch.compile(
                    model.encoder,
                    mode=COMPILE_MODE,
                    dynamic=COMPILE_DYNAMIC,
                    fullgraph=False,
                )
                logger.info(
                    "Compiled encoder: mode=%s, dynamic=%s",
                    COMPILE_MODE, COMPILE_DYNAMIC,
                )
            except Exception as e:
                logger.warning("torch.compile failed, using eager: %s", e)
                self._compiled_encoder = model.encoder
        else:
            self._compiled_encoder = model.encoder

    # ------------------------------------------------------------------
    # Prompt token caching
    # ------------------------------------------------------------------

    def _get_prompt_tokens(self, source_lang: str, target_lang: str, timestamps: bool = False) -> torch.Tensor:
        key = (source_lang, target_lang, timestamps)
        if key not in self._prompt_cache:
            turns = self.model.prompt.get_default_dialog_slots()
            for turn in turns:
                if turn["role"] == "user":
                    turn["slots"]["source_lang"] = f"<|{source_lang}|>"
                    turn["slots"]["target_lang"] = f"<|{target_lang}|>"
                    turn["slots"]["timestamp"] = "<|timestamp|>" if timestamps else "<|notimestamp|>"
            encoded = self.model.prompt.encode_dialog(turns=turns)
            self._prompt_cache[key] = encoded["context_ids"].to(self._device)
        return self._prompt_cache[key]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warmup(self):
        """Two-phase warmup for torch.compile + CUDA graphs.

        Phase 1: Run on main thread to trigger Dynamo trace + Inductor
                 compilation.  Populates the compiler cache.
        Phase 2: Re-run on _gpu_executor thread.  Dynamo/Inductor results
                 are cached so this is fast — it only captures CUDA graphs
                 on the correct thread.

        This avoids the ~60s-per-size penalty of cold-compiling on the GPU
        thread (which previously took 20+ minutes for 22 sizes).
        """
        if not COMPILE_ENCODER:
            return

        dummy_len = int(TARGET_SR * 3.5)
        dummy_audio = [np.zeros(dummy_len, dtype=np.float32)]

        # Build warmup sizes: powers-of-2 + neighbors
        warmup_sizes = set()
        bs = 1
        while bs <= self.max_batch_size:
            warmup_sizes.add(bs)
            if bs > 1:
                warmup_sizes.add(bs - 1)
            if bs + 1 <= self.max_batch_size:
                warmup_sizes.add(bs + 1)
            bs *= 2
        warmup_sizes.add(self.max_batch_size)
        warmup_sizes = sorted(warmup_sizes)

        # Phase 1: compile on main thread (populates Dynamo/Inductor cache)
        logger.info(
            "Phase 1: Compiling encoder for %d batch sizes (mode=%s) ...",
            len(warmup_sizes), COMPILE_MODE,
        )
        t0 = time.time()
        for bs in warmup_sizes:
            batch = dummy_audio * bs
            try:
                self._gpu_inference_sync(batch, "en", "en", False)
            except Exception as e:
                logger.warning("Phase 1 warmup failed bs=%d: %s", bs, e)
            if bs <= 2 or bs in (32, 64, 128, 256):
                logger.info("  phase1 bs=%d  (%.1fs)", bs, time.time() - t0)
        logger.info("Phase 1 done in %.1fs", time.time() - t0)

        # Phase 2: re-run on GPU executor thread (captures CUDA graphs)
        def _capture_graphs():
            logger.info("Phase 2: Capturing CUDA graphs on GPU thread ...")
            t1 = time.time()
            for bs in warmup_sizes:
                batch = dummy_audio * bs
                try:
                    self._gpu_inference_sync(batch, "en", "en", False)
                except Exception as e:
                    logger.warning("Phase 2 failed bs=%d: %s", bs, e)
            logger.info(
                "Phase 2 done: %d CUDA graphs in %.1fs",
                len(warmup_sizes), time.time() - t1,
            )

        future = _gpu_executor.submit(_capture_graphs)
        future.result()
        logger.info(
            "Warmup complete: %d sizes, total %.1fs",
            len(warmup_sizes), time.time() - t0,
        )

    def start(self, loop: asyncio.AbstractEventLoop = None):
        self._loop = loop or asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._task = self._loop.create_task(self._batch_loop())
        self._started = True
        logger.info(
            "In-flight batcher started (max_batch=%d, zero-wait, pipelined, "
            "pre-alloc=%d samples, gpu_executor=dedicated, audio_pool=32)",
            self.max_batch_size, self._max_audio_samples,
        )

    def stop(self):
        if self._task:
            self._task.cancel()
        self._started = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio_data: Union[str, bytes, np.ndarray],
        source_lang: str,
        target_lang: str,
        timestamps: bool = False,
    ):
        """Submit audio and wait for transcription result.

        Audio decoding happens on a CPU thread pool (parallel with GPU).
        The decoded numpy array is then queued for GPU inference.
        """
        # Decode audio on CPU thread pool — doesn't block event loop or GPU
        audio_np = await self._loop.run_in_executor(
            _audio_pool, _decode_audio, audio_data
        )

        future = self._loop.create_future()
        req = TranscribeRequest(
            audio=audio_np,
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
        """Transcribe a numpy audio buffer directly."""
        if len(audio_buffer) < int(TARGET_SR * 0.1):
            return ""
        try:
            result = await self.transcribe(audio_buffer, source_lang, target_lang)
            text = result[0].text if hasattr(result[0], "text") else str(result[0])
            return text.strip()
        except Exception as e:
            logger.error("Transcribe buffer error: %s", e)
            return ""

    # ------------------------------------------------------------------
    # In-flight batch loop (pipelined)
    # ------------------------------------------------------------------

    async def _batch_loop(self):
        """Continuous batch loop: zero-wait drain, pipelined GPU execution.

        While GPU processes batch N, the event loop is free to accept new
        requests and collect batch N+1.  This maximizes GPU utilization.
        """
        logger.info("In-flight batch loop running (pipelined)")
        while True:
            try:
                # Block until at least one item is available
                first = await self._queue.get()
                batch = [first]

                # Drain everything available RIGHT NOW — no waiting
                while len(batch) < self.max_batch_size:
                    try:
                        req = self._queue.get_nowait()
                        batch.append(req)
                    except asyncio.QueueEmpty:
                        break

                # Group by (source_lang, target_lang, timestamps)
                groups: dict[tuple, list[TranscribeRequest]] = {}
                for req in batch:
                    key = (req.source_lang, req.target_lang, req.timestamps)
                    groups.setdefault(key, []).append(req)

                for (src, tgt, ts), reqs in groups.items():
                    try:
                        # Run GPU inference on dedicated executor
                        # Event loop stays free → can accept new requests (pipeline)
                        results = await self._loop.run_in_executor(
                            _gpu_executor,
                            self._gpu_inference_sync,
                            [r.audio for r in reqs], src, tgt, ts,
                        )
                        for req, result in zip(reqs, results):
                            if not req.future.done():
                                req.future.set_result([result])
                    except Exception as e:
                        logger.error("Batch error: %s", e, exc_info=True)
                        for req in reqs:
                            if not req.future.done():
                                req.future.set_exception(e)

                if len(batch) > 1:
                    logger.debug("In-flight batch: %d items", len(batch))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Batch loop error: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # GPU inference (runs on dedicated GPU thread)
    # ------------------------------------------------------------------

    def _gpu_inference_sync(self, audio_arrays, source_lang, target_lang, timestamps):
        """Pad → preprocess → encode → decode on GPU.

        Uses pre-allocated buffers and non-blocking transfers.
        """
        lengths = [len(a) for a in audio_arrays]
        max_len = max(lengths)
        batch_size = len(audio_arrays)

        # Use pre-allocated buffer if audio fits, else allocate
        if max_len <= self._max_audio_samples and batch_size <= self.max_batch_size:
            # Zero only the region we'll use (faster than full buffer)
            audio_tensor = self._audio_buf[:batch_size, :max_len]
            audio_tensor.zero_()
            # Stage on pinned CPU memory, then transfer non-blocking
            cpu_stage = self._cpu_staging[:batch_size, :max_len]
            cpu_stage.zero_()
            for i, (arr, length) in enumerate(zip(audio_arrays, lengths)):
                cpu_stage[i, :length] = torch.from_numpy(arr)
            audio_tensor.copy_(cpu_stage, non_blocking=True)
            length_tensor = self._length_buf[:batch_size]
            length_tensor.copy_(
                torch.tensor(lengths, dtype=torch.long), non_blocking=True
            )
        else:
            # Fallback: allocate on the fly for oversized inputs
            audio_tensor = torch.zeros(
                batch_size, max_len, dtype=torch.float32, device=self._device
            )
            for i, (arr, length) in enumerate(zip(audio_arrays, lengths)):
                audio_tensor[i, :length] = torch.as_tensor(arr, device=self._device)
            length_tensor = torch.tensor(
                lengths, dtype=torch.long, device=self._device
            )

        with torch.inference_mode():
            processed, processed_len = self.model.preprocessor(
                input_signal=audio_tensor, length=length_tensor
            )
            encoded, encoded_len = self._compiled_encoder(
                audio_signal=processed, length=processed_len
            )
            enc_states = self.model.encoder_decoder_proj(encoded.permute(0, 2, 1))
            enc_mask = self._lens_to_mask(
                encoded_len, enc_states.shape[1]
            ).to(enc_states.dtype)

            if self.model.use_transf_encoder:
                enc_states = self.model.transf_encoder(
                    encoder_states=enc_states, encoder_mask=enc_mask
                )

            # Use <|timestamp|> prompt only for decoder-token fallback path;
            # CTC forced alignment doesn't need it (decoder just produces text)
            use_decoder_ts = timestamps and self._timestamps_asr_model is None
            prompt_ids = self._get_prompt_tokens(source_lang, target_lang, use_decoder_ts)
            # .expand() is zero-copy (no memory allocation), unlike .repeat()
            decoder_input_ids = prompt_ids.unsqueeze(0).expand(batch_size, -1)

            hypotheses = self.model.decoding.decode_predictions_tensor(
                encoder_hidden_states=enc_states,
                encoder_input_mask=enc_mask,
                decoder_input_ids=decoder_input_ids,
                return_hypotheses=timestamps,
            )

        # Timestamp post-processing
        if timestamps:
            fns = _get_timestamp_funcs()
            if self._timestamps_asr_model is not None:
                # Primary: CTC forced alignment (Viterbi) — most accurate
                hypotheses = fns["ctc_align"](
                    audio=[torch.from_numpy(a) for a in audio_arrays],
                    batch_size=batch_size,
                    external_ctc_model=self._timestamps_asr_model,
                    main_model_predictions=hypotheses,
                    timestamp_type=['word', 'segment'],
                    viterbi_device=self._device,
                )
            else:
                # Fallback: parse <|frame_number|> tokens from decoder output
                hypotheses = fns["aed_parse"](
                    hypotheses,
                    self._subsampling_factor,
                    self._window_stride,
                )
            # Both paths may return list-of-lists (beam); flatten
            if hypotheses and isinstance(hypotheses[0], list):
                hypotheses = [h[0] for h in hypotheses]

        results = []
        for hyp in hypotheses:
            if hasattr(hyp, "text"):
                text = hyp.text
                ts = getattr(hyp, "timestamp", None) if timestamps else None
            else:
                text = str(hyp)
                ts = None
            text = self._strip_special_tokens(text)
            if ts:
                ts = self._clean_timestamps(ts)
            results.append(SimpleHypothesis(text=text, timestamp=ts))
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    _SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")

    @classmethod
    def _strip_special_tokens(cls, text: str) -> str:
        return cls._SPECIAL_TOKEN_RE.sub("", text).strip()

    @classmethod
    def _clean_timestamps(cls, ts: dict) -> dict:
        """Remove entries that are purely special tokens from timestamp dicts."""
        cleaned = {}
        for key in ("word", "segment", "char"):
            items = ts.get(key)
            if not items:
                continue
            out = []
            for entry in items:
                text_key = key if key != "char" else "char"  # word/segment/char
                raw = entry.get(text_key, entry.get("word", entry.get("segment", "")))
                stripped = cls._SPECIAL_TOKEN_RE.sub("", raw).strip()
                if not stripped:
                    continue  # skip entries that are only special tokens
                entry = dict(entry)
                entry[text_key] = stripped
                out.append(entry)
            cleaned[key] = out
        return cleaned

    @staticmethod
    def _lens_to_mask(lens, max_length):
        batch_size = lens.shape[0]
        arange = torch.arange(max_length, device=lens.device)
        return arange.expand(batch_size, max_length) < lens.unsqueeze(1)
