#!/usr/bin/env python3
"""
Benchmark suite for the Canary 1B V2 service.

Measures:
  - REST /transcribe latency & RTFx for various audio durations
  - WebSocket streaming first-response latency & total latency
  - Concurrent request throughput
  - Translation vs ASR overhead
  - GPU memory usage
"""

import asyncio
import json
import os
import statistics
import sys
import tempfile
import time

import numpy as np
import requests
import soundfile as sf

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/stream"
SR = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_speech_audio(duration_s: float, sr: int = SR) -> str:
    """Generate a WAV file with speech-like formant audio."""
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    # Vowel-like formants with pitch modulation
    f0 = 120  # fundamental
    signal = (
        0.4 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 730 * t)
        + 0.15 * np.sin(2 * np.pi * 1090 * t)
        + 0.10 * np.sin(2 * np.pi * 2440 * t)
    )
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    signal = (signal * envelope * 0.3).astype(np.float32)
    path = tempfile.mktemp(suffix=".wav")
    sf.write(path, signal, sr)
    return path


def rest_transcribe(path: str, source_lang="en", target_lang="en", timestamps=False):
    """Call REST /transcribe and return (result, elapsed_s)."""
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/transcribe",
            files={"audio": ("audio.wav", f, "audio/wav")},
            data={
                "source_lang": source_lang,
                "target_lang": target_lang,
                "timestamps": str(timestamps).lower(),
            },
            timeout=120,
        )
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), elapsed


# ---------------------------------------------------------------------------
# Benchmark 1: REST latency vs audio duration
# ---------------------------------------------------------------------------

def bench_rest_latency():
    print("=" * 65)
    print("BENCHMARK 1: REST /transcribe — Latency vs Audio Duration")
    print("=" * 65)
    durations = [1, 2, 5, 10, 20, 30]
    results = []

    for dur in durations:
        path = generate_speech_audio(dur)
        times = []
        for trial in range(3):
            _, elapsed = rest_transcribe(path)
            times.append(elapsed)
        os.unlink(path)

        avg = statistics.mean(times)
        rtfx = dur / avg
        results.append((dur, avg, rtfx))
        print(f"  {dur:3d}s audio  →  avg {avg:.3f}s  RTFx {rtfx:7.1f}x  ({', '.join(f'{t:.3f}s' for t in times)})")

    print()
    avg_rtfx = statistics.mean(r[2] for r in results)
    print(f"  Average RTFx: {avg_rtfx:.1f}x realtime")
    return results


# ---------------------------------------------------------------------------
# Benchmark 2: REST translation overhead
# ---------------------------------------------------------------------------

def bench_translation_overhead():
    print()
    print("=" * 65)
    print("BENCHMARK 2: ASR vs Translation Overhead (10s audio)")
    print("=" * 65)
    path = generate_speech_audio(10)

    modes = [
        ("ASR (en→en)", "en", "en"),
        ("Translate (en→fr)", "en", "fr"),
        ("Translate (en→es)", "en", "es"),
        ("Translate (en→de)", "en", "de"),
    ]

    for label, src, tgt in modes:
        times = []
        for _ in range(3):
            _, elapsed = rest_transcribe(path, source_lang=src, target_lang=tgt)
            times.append(elapsed)
        avg = statistics.mean(times)
        print(f"  {label:25s}  avg {avg:.3f}s  (RTFx {10/avg:.1f}x)")

    os.unlink(path)


# ---------------------------------------------------------------------------
# Benchmark 3: REST with timestamps overhead
# ---------------------------------------------------------------------------

def bench_timestamps_overhead():
    print()
    print("=" * 65)
    print("BENCHMARK 3: Timestamps Overhead (10s audio)")
    print("=" * 65)
    path = generate_speech_audio(10)

    for ts_flag, label in [(False, "No timestamps"), (True, "With timestamps")]:
        times = []
        for _ in range(3):
            _, elapsed = rest_transcribe(path, timestamps=ts_flag)
            times.append(elapsed)
        avg = statistics.mean(times)
        print(f"  {label:20s}  avg {avg:.3f}s")

    os.unlink(path)


# ---------------------------------------------------------------------------
# Benchmark 4: WebSocket streaming latency
# ---------------------------------------------------------------------------

def bench_ws_streaming():
    print()
    print("=" * 65)
    print("BENCHMARK 4: WebSocket Streaming Latency")
    print("=" * 65)

    async def run_ws_bench(duration_s):
        chunk_ms = 80
        chunk_samples = int(SR * chunk_ms / 1000)
        n_chunks = int(duration_s * 1000 / chunk_ms)

        # Generate audio
        n = int(SR * duration_s)
        t = np.linspace(0, duration_s, n, endpoint=False)
        signal = (
            0.3 * np.sin(2 * np.pi * 730 * t)
            + 0.2 * np.sin(2 * np.pi * 1090 * t)
        )
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
        audio_f32 = (signal * envelope * 0.3).astype(np.float32)
        audio_int16 = (audio_f32 * 32768).clip(-32768, 32767).astype(np.int16)

        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({"source_lang": "en", "target_lang": "en"}))
            ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            assert ready["status"] == "ready"

            t_start = time.perf_counter()
            first_response_time = None
            all_responses = []

            async def send_audio():
                for i in range(n_chunks):
                    start = i * chunk_samples
                    end = start + chunk_samples
                    chunk = audio_int16[start:end]
                    if len(chunk) > 0:
                        await ws.send(chunk.tobytes())
                    await asyncio.sleep(chunk_ms / 1000)
                await asyncio.sleep(0.2)
                await ws.send(json.dumps({"action": "end_utterance"}))

            async def recv_responses():
                nonlocal first_response_time
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=8)
                        msg = json.loads(raw)
                        now = time.perf_counter()
                        if "text" in msg and first_response_time is None:
                            first_response_time = now - t_start
                        if "text" in msg:
                            all_responses.append((now - t_start, msg))
                        if msg.get("is_final"):
                            break
                    except asyncio.TimeoutError:
                        break

            await asyncio.gather(
                asyncio.create_task(send_audio()),
                asyncio.create_task(recv_responses()),
            )

            total_time = time.perf_counter() - t_start

        return {
            "duration_s": duration_s,
            "first_response_ms": first_response_time * 1000 if first_response_time else None,
            "total_time_s": total_time,
            "n_responses": len(all_responses),
            "final_text": all_responses[-1][1]["text"] if all_responses else "",
        }

    durations = [2, 5, 10]
    for dur in durations:
        results = []
        for _ in range(3):
            r = asyncio.run(run_ws_bench(dur))
            results.append(r)

        avg_first = statistics.mean(r["first_response_ms"] for r in results if r["first_response_ms"])
        avg_total = statistics.mean(r["total_time_s"] for r in results)
        avg_responses = statistics.mean(r["n_responses"] for r in results)

        print(f"  {dur}s audio:")
        print(f"    First response : {avg_first:7.0f}ms (avg of 3 runs)")
        print(f"    Total time     : {avg_total:7.2f}s")
        print(f"    Responses      : {avg_responses:.1f} avg")
        print()


# ---------------------------------------------------------------------------
# Benchmark 5: Concurrent REST requests
# ---------------------------------------------------------------------------

def bench_concurrent():
    print("=" * 65)
    print("BENCHMARK 5: Concurrent REST Requests (5s audio)")
    print("=" * 65)

    path = generate_speech_audio(5)

    import concurrent.futures

    for n_workers in [1, 2, 4]:
        def do_request(_):
            return rest_transcribe(path)

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(do_request, i) for i in range(n_workers)]
            results = [f.result() for f in futures]
        wall_time = time.perf_counter() - t0

        individual_times = [r[1] for r in results]
        total_audio = 5 * n_workers
        throughput_rtfx = total_audio / wall_time

        print(f"  {n_workers} concurrent:")
        print(f"    Wall time      : {wall_time:.2f}s")
        print(f"    Individual     : {', '.join(f'{t:.2f}s' for t in individual_times)}")
        print(f"    Throughput RTFx: {throughput_rtfx:.1f}x  ({total_audio}s audio in {wall_time:.2f}s)")
        print()

    os.unlink(path)


# ---------------------------------------------------------------------------
# Benchmark 6: GPU memory usage
# ---------------------------------------------------------------------------

def bench_gpu_memory():
    print("=" * 65)
    print("BENCHMARK 6: GPU Memory Usage")
    print("=" * 65)
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Allocated : {allocated:.2f} GB")
            print(f"  Reserved  : {reserved:.2f} GB")
            print(f"  Total VRAM: {total:.1f} GB")
            print(f"  Free      : {total - reserved:.1f} GB")
        else:
            print("  No CUDA available (running on CPU)")
    except Exception as e:
        # Server runs in separate process, read nvidia-smi instead
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            used, total, free = result.stdout.strip().split(", ")
            print(f"  Used  : {int(used)/1024:.2f} GB")
            print(f"  Total : {int(total)/1024:.1f} GB")
            print(f"  Free  : {int(free)/1024:.1f} GB")
        else:
            print(f"  Could not query GPU: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("  Canary 1B V2 — Performance Benchmark")
    print("  GPU: NVIDIA L40S | NeMo 2.6.2 | FP16")
    print()

    bench_gpu_memory()
    print()
    bench_rest_latency()
    bench_translation_overhead()
    bench_timestamps_overhead()
    bench_ws_streaming()
    bench_concurrent()

    print("=" * 65)
    print("BENCHMARK COMPLETE")
    print("=" * 65)
