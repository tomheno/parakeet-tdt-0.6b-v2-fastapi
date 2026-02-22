#!/usr/bin/env python3
"""
Benchmark: Timestamps ON vs OFF for Canary 1B V2.

Tests both /transcribe (multipart) and /transcribe/raw (binary body) endpoints.

Usage:
    python benchmark_timestamps.py
"""

import asyncio
import os
import statistics
import time

import aiohttp
import soundfile as sf

BASE_URL = "http://localhost:8000"
SAMPLES_DIR = "/teamspace/studios/this_studio/samples"

# Use all available sample files
SAMPLE_FILES = [
    os.path.join(SAMPLES_DIR, f)
    for f in sorted(os.listdir(SAMPLES_DIR))
    if f.endswith((".wav", ".flac", ".mp3"))
]

# Measure actual durations
SAMPLE_DURATIONS = {}
for _p in SAMPLE_FILES:
    _info = sf.info(_p)
    SAMPLE_DURATIONS[_p] = _info.duration
AVG_DURATION = sum(SAMPLE_DURATIONS.values()) / len(SAMPLE_DURATIONS) if SAMPLE_DURATIONS else 3.0


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

async def transcribe_multipart(session, audio_bytes, timestamps: bool):
    """Single /transcribe request (multipart form)."""
    data = aiohttp.FormData()
    data.add_field("audio", audio_bytes, filename="audio.wav", content_type="audio/wav")
    data.add_field("source_lang", "en")
    data.add_field("target_lang", "en")
    data.add_field("timestamps", str(timestamps).lower())

    t0 = time.perf_counter()
    async with session.post(f"{BASE_URL}/transcribe", data=data) as resp:
        result = await resp.json()
        latency = (time.perf_counter() - t0) * 1000
    return result, latency


async def transcribe_raw(session, audio_bytes, timestamps: bool):
    """Single /transcribe/raw request (binary body, no multipart overhead)."""
    params = {"source_lang": "en", "target_lang": "en", "timestamps": str(timestamps).lower()}

    t0 = time.perf_counter()
    async with session.post(
        f"{BASE_URL}/transcribe/raw",
        data=audio_bytes,
        params=params,
        headers={"Content-Type": "audio/wav"},
    ) as resp:
        result = await resp.json()
        latency = (time.perf_counter() - t0) * 1000
    return result, latency


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def _stats(latencies):
    latencies.sort()
    n = len(latencies)
    return {
        "avg": statistics.mean(latencies),
        "p50": latencies[n // 2],
        "p95": latencies[int(n * 0.95)],
        "p99": latencies[int(n * 0.99)],
        "min": min(latencies),
        "max": max(latencies),
    }


async def burst_benchmark(audio_bytes_list, concurrency, timestamps: bool, label: str, *, use_raw=False):
    """Run burst of concurrent requests and report stats."""
    tasks_data = [audio_bytes_list[i % len(audio_bytes_list)] for i in range(concurrency)]
    fn = transcribe_raw if use_raw else transcribe_multipart

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        await fn(session, tasks_data[0], timestamps)

        t0 = time.perf_counter()
        tasks = [fn(session, ab, timestamps) for ab in tasks_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall = time.perf_counter() - t0

    latencies = []
    errors = 0
    for r in results:
        if isinstance(r, Exception):
            errors += 1
        else:
            _, lat = r
            latencies.append(lat)

    if not latencies:
        print(f"  {label}: ALL ERRORS ({errors})")
        return None

    s = _stats(latencies)
    total_audio = concurrency * AVG_DURATION
    rtfx = total_audio / wall

    print(f"  {label}: n={concurrency}  wall={wall:.2f}s  RTFx={rtfx:.0f}")
    print(f"    Avg={s['avg']:.0f}ms  P50={s['p50']:.0f}ms  P95={s['p95']:.0f}ms  P99={s['p99']:.0f}ms  "
          f"Min={s['min']:.0f}ms  Max={s['max']:.0f}ms  Errors={errors}")
    return {"avg": s["avg"], "p50": s["p50"], "p95": s["p95"], "p99": s["p99"], "wall": wall, "rtfx": rtfx}


async def single_latency_test(audio_bytes_list, timestamps: bool, label: str, *, use_raw=False, n_runs: int = 10):
    """Single-request latency test (sequential, no concurrency)."""
    fn = transcribe_raw if use_raw else transcribe_multipart

    async with aiohttp.ClientSession() as session:
        await fn(session, audio_bytes_list[0], timestamps)
        await fn(session, audio_bytes_list[0], timestamps)

        latencies = []
        sample_result = None
        for i in range(n_runs):
            ab = audio_bytes_list[i % len(audio_bytes_list)]
            result, lat = await fn(session, ab, timestamps)
            latencies.append(lat)
            if i == 0:
                sample_result = result

    s = _stats(latencies)
    print(f"  {label}: n={n_runs}  Avg={s['avg']:.0f}ms  P50={s['p50']:.0f}ms  P95={s['p95']:.0f}ms  "
          f"Min={s['min']:.0f}ms  Max={s['max']:.0f}ms")

    return {"avg": s["avg"], "p50": s["p50"], "sample": sample_result}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_endpoint_bench(audio_bytes_list, *, use_raw: bool):
    ep = "/transcribe/raw" if use_raw else "/transcribe"
    tag = "RAW" if use_raw else "MULTIPART"

    print(f"\n{'=' * 70}")
    print(f"  {tag} endpoint: {ep}")
    print(f"{'=' * 70}")

    # ---- Single-request latency ----
    print(f"\n  1. Single-request latency (sequential, 10 runs)")
    print(f"  {'-' * 60}")
    off_s = await single_latency_test(audio_bytes_list, False, "ts=OFF", use_raw=use_raw)
    on_s = await single_latency_test(audio_bytes_list, True, "ts=ON ", use_raw=use_raw)

    overhead_ms = on_s["avg"] - off_s["avg"]
    overhead_pct = (overhead_ms / off_s["avg"]) * 100
    print(f"\n  Timestamp overhead: +{overhead_ms:.0f}ms (+{overhead_pct:.1f}%)")

    # ---- Burst ----
    print(f"\n  2. Burst benchmark — Timestamps OFF vs ON")
    print(f"  {'-' * 60}")

    concurrencies = [1, 4, 16, 32, 64, 128, 256, 512]

    for c in concurrencies:
        print(f"\n  --- Concurrency {c} ---")
        off = await burst_benchmark(audio_bytes_list, c, False, "OFF", use_raw=use_raw)
        on = await burst_benchmark(audio_bytes_list, c, True, " ON", use_raw=use_raw)
        if off and on:
            overhead_pct = ((on["avg"] - off["avg"]) / off["avg"]) * 100
            print(f"    Delta: Avg latency +{overhead_pct:.1f}%  RTFx {on['rtfx'] - off['rtfx']:+.0f}")


async def main():
    print("=" * 70)
    print("Canary 1B V2 — Timestamps Benchmark (Multipart vs Raw)")
    print("=" * 70)

    # Load audio files
    audio_bytes_list = []
    for path in SAMPLE_FILES:
        with open(path, "rb") as f:
            audio_bytes_list.append(f.read())
    print(f"\nLoaded {len(audio_bytes_list)} sample files")
    for p in SAMPLE_FILES:
        print(f"  - {os.path.basename(p):30s} {SAMPLE_DURATIONS[p]:.1f}s")
    print(f"  Average duration: {AVG_DURATION:.2f}s")

    # Run both endpoints
    await run_endpoint_bench(audio_bytes_list, use_raw=False)
    await run_endpoint_bench(audio_bytes_list, use_raw=True)

    print(f"\n{'=' * 70}")
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
