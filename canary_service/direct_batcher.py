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
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch

from .config import TARGET_SR, logger
from .optimizations import _env_bool

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
# Encoder acceleration config
# ---------------------------------------------------------------------------

# CUDA_GRAPH_ENCODER=1 — manual CUDA graph capture for encoder (default ON)
#   Fast to capture (seconds), replays captured graphs for matching shapes.
#   Captures graphs for batch sizes: 1,2,4,8,16,32,64,128,256.
#   Falls back to eager for uncaptured shapes.
#
# COMPILE_ENCODER=0 — torch.compile alternative (default OFF, slow compile)
CUDA_GRAPH_ENCODER = _env_bool("CUDA_GRAPH_ENCODER", default=False)
COMPILE_ENCODER = _env_bool("COMPILE_ENCODER", default=False)
COMPILE_MODE = os.getenv("COMPILE_MODE", "default")
COMPILE_DYNAMIC = _env_bool("COMPILE_DYNAMIC", default=True)

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

        # Encoder acceleration: CUDA graphs (fast) or torch.compile (slow)
        self._compiled_encoder = model.encoder
        # {(batch_size, audio_len): (graph, static_audio, static_lengths, static_encoded, static_encoded_len)}
        self._encoder_graphs = {}
        # Sorted list of captured audio lengths for quick lookup
        self._graph_audio_lens = []

        if COMPILE_ENCODER and not CUDA_GRAPH_ENCODER:
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

    # ------------------------------------------------------------------
    # CUDA graph encoder
    # ------------------------------------------------------------------

    def _capture_encoder_graph(self, batch_size: int, max_audio_len: int):
        """Capture a CUDA graph of preprocessor + encoder for a fixed shape.

        Returns (graph, static_processed, static_processed_len,
                 static_encoded, static_encoded_len).
        """
        # Static input buffers (fixed shape, reused across replays)
        static_audio = torch.zeros(
            batch_size, max_audio_len, dtype=torch.float32, device=self._device
        )
        static_lengths = torch.full(
            (batch_size,), max_audio_len, dtype=torch.long, device=self._device
        )

        # Warmup runs (mandatory before capture to stabilize cuDNN etc.)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                processed, processed_len = self.model.preprocessor(
                    input_signal=static_audio, length=static_lengths
                )
                encoded, encoded_len = self.model.encoder(
                    audio_signal=processed, length=processed_len
                )
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            static_processed, static_processed_len = self.model.preprocessor(
                input_signal=static_audio, length=static_lengths
            )
            static_encoded, static_encoded_len = self.model.encoder(
                audio_signal=static_processed, length=static_processed_len
            )

        return (graph, static_audio, static_lengths,
                static_encoded, static_encoded_len)

    def _run_encoder_graph(self, audio_tensor, length_tensor):
        """Run encoder using CUDA graph if available, else eager.

        Finds the smallest captured graph whose audio length >= actual max_len,
        at the matching batch size. This minimizes wasted padding compute.
        """
        bs = audio_tensor.shape[0]
        max_len = audio_tensor.shape[1]

        # Find smallest captured audio length >= max_len for this batch size
        for graph_len in self._graph_audio_lens:
            if graph_len >= max_len:
                entry = self._encoder_graphs.get((bs, graph_len))
                if entry is not None:
                    graph, static_audio, static_lengths, static_encoded, static_encoded_len = entry
                    # Copy audio into static buffer
                    static_audio[:bs, :max_len].copy_(audio_tensor, non_blocking=True)
                    if max_len < graph_len:
                        static_audio[:bs, max_len:graph_len].zero_()
                    static_lengths.copy_(length_tensor, non_blocking=True)
                    graph.replay()
                    return static_encoded.clone(), static_encoded_len.clone()

        # Eager fallback (audio too long or batch size not captured)
        with torch.inference_mode():
            processed, processed_len = self.model.preprocessor(
                input_signal=audio_tensor, length=length_tensor
            )
            encoded, encoded_len = self._compiled_encoder(
                audio_signal=processed, length=processed_len
            )
        return encoded, encoded_len

    def warmup(self):
        """Warmup: CUDA graph capture or torch.compile warmup on GPU thread.

        For CUDA graphs: captures graphs for power-of-2 batch sizes at a
        representative audio length (~5s). Fast — takes ~30s total.
        """
        if not CUDA_GRAPH_ENCODER and not COMPILE_ENCODER:
            return

        def _warmup_on_gpu():
            t0 = time.time()

            if CUDA_GRAPH_ENCODER:
                # Capture CUDA graphs for common (batch_size, audio_length) combos.
                # Multiple audio lengths to minimize padding waste:
                #   2s  — short utterances
                #   5s  — typical speech
                #   10s — longer utterances
                #   20s — long audio
                audio_lens = [TARGET_SR * d for d in (2, 5, 10, 20)]

                # Power-of-2 batch sizes up to max
                capture_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
                if self.max_batch_size not in capture_sizes and self.max_batch_size <= 256:
                    capture_sizes.append(self.max_batch_size)
                capture_sizes = sorted(s for s in capture_sizes if s <= self.max_batch_size)

                total = len(capture_sizes) * len(audio_lens)
                logger.info(
                    "CUDA graph encoder warmup: %d batch sizes × %d audio lengths = %d graphs ...",
                    len(capture_sizes), len(audio_lens), total,
                )
                captured = 0
                for audio_len in audio_lens:
                    for bs in capture_sizes:
                        try:
                            entry = self._capture_encoder_graph(bs, audio_len)
                            self._encoder_graphs[(bs, audio_len)] = entry
                            captured += 1
                        except Exception as e:
                            logger.warning("  bs=%d len=%d capture failed: %s", bs, audio_len, e)
                    logger.info(
                        "  audio=%ds: %d/%d batch sizes captured (%.1fs)",
                        audio_len // TARGET_SR, len(capture_sizes),
                        len(capture_sizes), time.time() - t0,
                    )

                # Store sorted audio lengths for quick lookup
                self._graph_audio_lens = sorted(set(
                    k[1] for k in self._encoder_graphs.keys()
                ))

                # Verify replay
                logger.info("Verifying graph replay ...")
                dummy = np.zeros(int(TARGET_SR * 3.0), dtype=np.float32)
                try:
                    self._gpu_inference_sync([dummy], "en", "en", False)
                    logger.info("  Verify OK")
                except Exception as e:
                    logger.warning("  Verify failed: %s — disabling CUDA graph encoder", e)
                    self._encoder_graphs.clear()
                    self._graph_audio_lens.clear()

                logger.info(
                    "CUDA graph encoder ready: %d/%d graphs in %.1fs (audio lens: %s)",
                    captured, total, time.time() - t0,
                    [l // TARGET_SR for l in self._graph_audio_lens],
                )

            elif COMPILE_ENCODER:
                # torch.compile warmup (slow but comprehensive)
                warmup_durations = [d / 2.0 for d in range(2, 21)]
                dummy_audios = {
                    dur: [np.zeros(int(TARGET_SR * dur), dtype=np.float32)]
                    for dur in warmup_durations
                }
                logger.info("torch.compile warmup: %d audio lengths ...", len(warmup_durations))
                for dur in warmup_durations:
                    try:
                        self._gpu_inference_sync(dummy_audios[dur], "en", "en", False)
                    except Exception as e:
                        logger.warning("Warmup dur=%.1fs failed: %s", dur, e)
                logger.info("torch.compile warmup done (%.1fs)", time.time() - t0)

        future = _gpu_executor.submit(_warmup_on_gpu)
        future.result()

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
            if self._encoder_graphs:
                # CUDA graph path: preprocessor + encoder fused in graph
                encoded, encoded_len = self._run_encoder_graph(
                    audio_tensor, length_tensor
                )
            else:
                # Eager / torch.compile path
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

            # Invalidate cross-attention K/V cache before new decode
            from .optimizations import clear_kv_cache
            clear_kv_cache()

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
