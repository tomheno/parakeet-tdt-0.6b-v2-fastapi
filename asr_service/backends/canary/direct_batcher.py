"""
In-flight (continuous) batcher for Canary 1B V2.

Inspired by vLLM's continuous batching: no artificial wait — the GPU always
processes the maximum available batch.  Requests join immediately as they
arrive; completed requests return without waiting for the whole batch.

Pipeline (resolve-during-GPU):
  1. Audio arrives as bytes → decoded to numpy on CPU (parallel with GPU)
  2. Queued as ready numpy arrays
  3. Batch loop drains queue instantly (no wait if items available)
  4. GPU inference submitted on dedicated thread
  5. While GPU runs, previous batch results are resolved (overlap)
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

import ctypes
import pathlib

import numpy as np
import soundfile as sf
import torch

from .config import TARGET_SR, logger
from .optimizations import _env_bool

# ---------------------------------------------------------------------------
# GIL-free batch padding via C extension
# ---------------------------------------------------------------------------
# ctypes releases the GIL for the entire C call — one GIL release for the
# whole batch, not 256 individual numpy assignments that each fight for GIL
# with 64 audio decode threads.

_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_float_pp = ctypes.POINTER(_c_float_p)
_c_int_p = ctypes.POINTER(ctypes.c_int)

_c_longlong_p = ctypes.POINTER(ctypes.c_longlong)

_so_path = pathlib.Path(__file__).with_name("_fast_copy.so")
if _so_path.exists():
    _fast_lib = ctypes.CDLL(str(_so_path))
    _fast_lib.batch_pad_copy.restype = None
    _fast_lib.batch_pad_copy.argtypes = [_c_float_p, ctypes.c_int,
                                          _c_float_pp, _c_int_p, ctypes.c_int]
    _fast_lib.batch_pad_scatter.restype = None
    _fast_lib.batch_pad_scatter.argtypes = [_c_float_p, ctypes.c_int,
                                             _c_float_p, _c_longlong_p,
                                             _c_int_p, ctypes.c_int]
    logger.info("Loaded _fast_copy.so — GIL-free batch padding enabled")
else:
    _fast_lib = None
    logger.warning("_fast_copy.so not found — falling back to numpy loop")

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

MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "256"))

# Thread pool for CPU-bound audio decoding (runs in parallel with GPU).
# Fewer threads = less GIL contention with GPU thread.
# At 500 RPS × ~2ms decode = 1s of work/s → 16 threads is plenty.
_AUDIO_WORKERS = int(os.getenv("AUDIO_WORKERS", "16"))
_audio_pool = ThreadPoolExecutor(max_workers=_AUDIO_WORKERS)

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
    beam_size: int = 0          # 0 = greedy (default), >1 = beam search
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

        # Max audio length for pre-allocated buffers
        self._max_audio_samples = TARGET_SR * 30  # 30s max

        # Double-buffered pinned CPU staging for prep/GPU overlap.
        # Buffer A and B alternate: prep fills one while GPU reads the other.
        self._cpu_staging = [
            torch.zeros(max_batch_size, self._max_audio_samples,
                        dtype=torch.float32, pin_memory=True)
            for _ in range(2)
        ]
        self._cpu_staging_np = [buf.numpy() for buf in self._cpu_staging]
        self._gpu_bufs = [
            torch.zeros(max_batch_size, self._max_audio_samples,
                        dtype=torch.float32, device=self._device)
            for _ in range(2)
        ]
        self._length_bufs = [
            torch.zeros(max_batch_size, dtype=torch.long, device=self._device)
            for _ in range(2)
        ]
        self._buf_idx = 0  # alternates 0/1

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
            "In-flight batcher started (max_batch=%d, zero-wait, "
            "pre-alloc=%d samples, gpu_executor=dedicated, audio_pool=64)",
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
        beam_size: int = 0,
    ):
        """Submit audio and wait for transcription result.

        Audio decoding happens on a CPU thread pool (parallel with GPU).
        The decoded numpy array is then queued for GPU inference.

        beam_size: 0 = greedy (default, fastest), >1 = beam search
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
            beam_size=beam_size,
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
        """Batch loop with resolve-during-GPU overlap.

        Submit GPU → resolve previous batch (overlap) → await GPU.
        """
        logger.info("In-flight batch loop running (resolve-during-GPU pipeline)")
        _batch_count = 0
        _timing_accum = 0.0
        _timing_batches = 0
        _timing_items = 0
        _last_gpu_end = 0.0

        pending_resolve: list[tuple] = []

        while True:
            try:
                first = await self._queue.get()
                batch = [first]
                while len(batch) < self.max_batch_size:
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                groups: dict[tuple, list[TranscribeRequest]] = {}
                for req in batch:
                    key = (req.source_lang, req.target_lang, req.timestamps, req.beam_size)
                    groups.setdefault(key, []).append(req)

                for (src, tgt, ts, beam), reqs in groups.items():
                    try:
                        gpu_future = self._loop.run_in_executor(
                            _gpu_executor,
                            self._gpu_inference_sync,
                            [r.audio for r in reqs], src, tgt, ts, beam,
                        )

                        if pending_resolve:
                            for preq, presult in pending_resolve:
                                if not preq.future.done():
                                    preq.future.set_result([presult])
                            pending_resolve.clear()

                        results = await gpu_future
                        t_gpu_done = time.monotonic()

                        for req, result in zip(reqs, results):
                            pending_resolve.append((req, result))

                        if _last_gpu_end > 0:
                            _timing_accum += t_gpu_done - _last_gpu_end
                            _timing_batches += 1
                            _timing_items += len(reqs)
                        _last_gpu_end = t_gpu_done

                    except Exception as e:
                        logger.error("Batch error: %s", e, exc_info=True)
                        for req in reqs:
                            if not req.future.done():
                                req.future.set_exception(e)

                if self._queue.empty() and pending_resolve:
                    for preq, presult in pending_resolve:
                        if not preq.future.done():
                            preq.future.set_result([presult])
                    pending_resolve.clear()

                _batch_count += 1
                if _batch_count % 20 == 0 and _timing_batches > 0:
                    n = _timing_batches
                    cycle = _timing_accum / n * 1000
                    eff = _timing_items / _timing_accum if _timing_accum > 0 else 0
                    logger.warning(
                        "BATCH TIMING (%d batches, %d items): cycle=%.1fms | qd=%d eff_rps=%.0f",
                        n, _timing_items, cycle, self._queue.qsize(), eff,
                    )
                    _timing_accum = 0.0
                    _timing_batches = 0
                    _timing_items = 0

            except asyncio.CancelledError:
                for preq, presult in pending_resolve:
                    if not preq.future.done():
                        preq.future.set_result([presult])
                break
            except Exception as e:
                logger.error("Batch loop error: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # GPU inference — sequential encode + decode on CUDA streams
    # ------------------------------------------------------------------

    def _prep_and_transfer(self, audio_arrays):
        """Pad audio into pinned buffer and transfer to GPU.

        Uses C extension (GIL-free) for the padding loop. One GIL release
        for the entire batch instead of 256 numpy assignments that each
        contend for the GIL with audio decode threads.
        """
        lengths = [len(a) for a in audio_arrays]
        max_len = max(lengths)
        batch_size = len(audio_arrays)
        buf_idx = self._buf_idx
        self._buf_idx ^= 1

        if max_len <= self._max_audio_samples and batch_size <= self.max_batch_size:
            padded_np = self._cpu_staging_np[buf_idx][:batch_size, :max_len]

            if _fast_lib is not None:
                # C extension: single GIL-free call for zero + copy.
                # Pointer array building is a fast Python loop (~0.5ms),
                # then batch_pad_copy runs entirely without GIL.
                PtrArr = (_c_float_p * batch_size)()
                for i, arr in enumerate(audio_arrays):
                    PtrArr[i] = arr.ctypes.data_as(_c_float_p)
                LenArr = (ctypes.c_int * batch_size)(*lengths)
                dst_ptr = padded_np.ctypes.data_as(_c_float_p)
                _fast_lib.batch_pad_copy(dst_ptr, max_len, PtrArr, LenArr, batch_size)
            else:
                # Fallback: numpy loop
                padded_np[:] = 0
                for i, (arr, length) in enumerate(zip(audio_arrays, lengths)):
                    padded_np[i, :length] = arr

            gpu_buf = self._gpu_bufs[buf_idx]
            gpu_buf[:batch_size, :max_len].copy_(
                self._cpu_staging[buf_idx][:batch_size, :max_len], non_blocking=True
            )
            length_buf = self._length_bufs[buf_idx]
            length_buf[:batch_size].copy_(
                torch.tensor(lengths, dtype=torch.long), non_blocking=True
            )
            audio_tensor = gpu_buf[:batch_size, :max_len]
            length_tensor = length_buf[:batch_size]
        else:
            padded_np = np.zeros((batch_size, max_len), dtype=np.float32)
            for i, (arr, length) in enumerate(zip(audio_arrays, lengths)):
                padded_np[i, :length] = arr
            audio_tensor = torch.from_numpy(padded_np).to(self._device, non_blocking=True)
            length_tensor = torch.tensor(
                lengths, dtype=torch.long, device=self._device
            )
        return audio_tensor, length_tensor, batch_size, lengths

    def _gpu_inference_sync(self, audio_arrays, source_lang, target_lang, timestamps, beam_size=0):
        """Prep + encode + decode on default CUDA stream."""
        import gc as _gc
        _gc_was_enabled = _gc.isenabled()
        _gc.disable()

        t_start = time.monotonic()

        # Switch decoding strategy if dynamic switching is available
        from .optimizations import _decoding_switch
        if _decoding_switch is not None:
            _decoding_switch(beam_size)

        audio_tensor, length_tensor, batch_size, lengths = \
            self._prep_and_transfer(audio_arrays)

        t_prep = time.monotonic()

        with torch.inference_mode():
            from .optimizations import clear_kv_cache

            if self._encoder_graphs:
                encoded, encoded_len = self._run_encoder_graph(
                    audio_tensor, length_tensor
                )
                torch.cuda.synchronize()
                t_preproc = t_prep
                t_enc = time.monotonic()
            else:
                processed, processed_len = self.model.preprocessor(
                    input_signal=audio_tensor, length=length_tensor
                )
                torch.cuda.synchronize()
                t_preproc = time.monotonic()
                encoded, encoded_len = self._compiled_encoder(
                    audio_signal=processed, length=processed_len
                )
                torch.cuda.synchronize()
                t_enc = time.monotonic()

            enc_states = self.model.encoder_decoder_proj(encoded.permute(0, 2, 1))
            enc_mask = self._lens_to_mask(
                encoded_len, enc_states.shape[1]
            ).to(enc_states.dtype)

            if self.model.use_transf_encoder:
                enc_states = self.model.transf_encoder(
                    encoder_states=enc_states, encoder_mask=enc_mask
                )

            use_decoder_ts = timestamps and self._timestamps_asr_model is None
            prompt_ids = self._get_prompt_tokens(source_lang, target_lang, use_decoder_ts)
            decoder_input_ids = prompt_ids.unsqueeze(0).expand(batch_size, -1)

            clear_kv_cache()
            hypotheses = self.model.decoding.decode_predictions_tensor(
                encoder_hidden_states=enc_states,
                encoder_input_mask=enc_mask,
                decoder_input_ids=decoder_input_ids,
                return_hypotheses=timestamps,
            )
            torch.cuda.synchronize()

            # Read actual step count from greedy generator
            _gs = getattr(self.model.decoding, 'decoding', None)
            _gs = getattr(_gs, 'greedy_search', None) if _gs else None
            _dec_actual = getattr(_gs, '_last_actual_steps', -1) if _gs else -1
            _dec_max_gen = getattr(_gs, '_last_max_gen', -1) if _gs else -1

        t_end = time.monotonic()

        if timestamps:
            fns = _get_timestamp_funcs()
            if self._timestamps_asr_model is not None:
                hypotheses = fns["ctc_align"](
                    audio=[torch.from_numpy(a) for a in audio_arrays],
                    batch_size=batch_size,
                    external_ctc_model=self._timestamps_asr_model,
                    main_model_predictions=hypotheses,
                    timestamp_type=['word', 'segment'],
                    viterbi_device=self._device,
                )
            else:
                hypotheses = fns["aed_parse"](
                    hypotheses, self._subsampling_factor, self._window_stride,
                )
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

        # Timing
        if not hasattr(self, '_gpu_timing_accum'):
            self._gpu_timing_accum = {
                "prep": 0.0, "preproc": 0.0, "enc": 0.0,
                "dec": 0.0, "n": 0, "items": 0,
                "actual_steps": 0, "max_gen": 0,
            }
        a = self._gpu_timing_accum
        a["prep"] += t_prep - t_start
        a["preproc"] += t_preproc - t_prep
        a["enc"] += t_enc - t_preproc
        a["dec"] += t_end - t_enc
        a["n"] += 1
        a["items"] += batch_size
        a["actual_steps"] += _dec_actual
        a["max_gen"] += _dec_max_gen
        if a["n"] % 20 == 0:
            n = a["n"]
            total = a["prep"] + a["preproc"] + a["enc"] + a["dec"]
            dec_ms = a["dec"] / n * 1000
            avg_steps = a["actual_steps"] / n
            avg_max = a["max_gen"] / n
            ms_per_step = dec_ms / avg_steps if avg_steps > 0 else 0
            logger.warning(
                "GPU TIMING (%d batches, %d items): "
                "prep=%.1fms mel=%.1fms enc=%.1fms dec=%.1fms "
                "[%.0f/%.0f steps, %.1fms/step] | total=%.1fms",
                n, a["items"],
                a["prep"] / n * 1000,
                a["preproc"] / n * 1000,
                a["enc"] / n * 1000,
                dec_ms,
                avg_steps, avg_max, ms_per_step,
                total / n * 1000,
            )
            self._gpu_timing_accum = {
                "prep": 0.0, "preproc": 0.0, "enc": 0.0,
                "dec": 0.0, "n": 0, "items": 0,
                "actual_steps": 0, "max_gen": 0,
            }

        if _gc_was_enabled:
            _gc.enable()
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
