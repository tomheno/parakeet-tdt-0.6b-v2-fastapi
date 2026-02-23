#!/usr/bin/env python3
"""
Call center ASR load test — simulates realistic conversation patterns.

Models 100-1000+ concurrent callers, each with natural think times between
ASR turns (user speaks → ASR → system responds → user thinks → speaks again).

User profiles:
  CallerRealistic (60%): 5-12s think time (typical conversation turn)
  CallerFast      (30%): 2-5s think time  (quick confirmations, IVR navigation)
  CallerBurst     (10%): 0.5-1.5s think   (rapid dictation, reading lists)

Usage:
    locust -f locustfile_callcenter.py \
        --host http://localhost:8000 \
        --headless -u 300 -r 30 --run-time 120s

    # Environment variables:
    #   SAMPLES_DIR   - Audio samples directory (default: /teamspace/studios/this_studio/samples)
    #   SOURCE_LANG   - Source language (default: en)
    #   TARGET_LANG   - Target language (default: en)
    #   RESULTS_DIR   - Where to save summary JSON (default: results)
"""

import json
import os
import random
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from locust import HttpUser, task, between, events


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLES_DIR = Path(os.getenv(
    "SAMPLES_DIR",
    "/teamspace/studios/this_studio/samples"))
SOURCE_LANG = os.getenv("SOURCE_LANG", "en")
TARGET_LANG = os.getenv("TARGET_LANG", "en")
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

_AUDIO_SAMPLES = []  # list of (filename, bytes, duration_s)


def _load_samples():
    """Load all WAV files from SAMPLES_DIR into memory."""
    global _AUDIO_SAMPLES
    if _AUDIO_SAMPLES:
        return

    for f in sorted(SAMPLES_DIR.iterdir()):
        if f.suffix.lower() in (".wav", ".flac", ".mp3"):
            audio_bytes = f.read_bytes()
            # Get duration from audio metadata
            try:
                info = sf.info(str(f))
                duration_s = info.duration
            except Exception:
                # Fallback: estimate from file size (16-bit PCM WAV)
                pcm_bytes = max(0, len(audio_bytes) - 44)
                duration_s = pcm_bytes / (SAMPLE_RATE * 2)

            _AUDIO_SAMPLES.append((f.name, audio_bytes, duration_s))

    if not _AUDIO_SAMPLES:
        raise RuntimeError(f"No audio files found in {SAMPLES_DIR}")


# ---------------------------------------------------------------------------
# Metrics collection (thread-safe)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
RTF_SAMPLES = []
LATENCY_SAMPLES = []
AUDIO_DURATIONS = []
TEXTS = []  # store a few transcriptions for spot-checking


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@events.init.add_listener
def on_init(environment, **kwargs):
    _load_samples()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    durations = [d for _, _, d in _AUDIO_SAMPLES]
    print(f"[INFO] Call center ASR benchmark")
    print(f"[INFO] Audio samples: {len(_AUDIO_SAMPLES)} files from {SAMPLES_DIR}")
    print(f"[INFO] Audio durations: {min(durations):.1f}s - {max(durations):.1f}s "
          f"(avg {np.mean(durations):.1f}s)")
    print(f"[INFO] Language: {SOURCE_LANG} → {TARGET_LANG}")
    print(f"[INFO] Profiles: Realistic (60%, 5-12s), Fast (30%, 2-5s), Burst (10%, 0.5-1.5s)")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats
    total = stats.total

    print("\n" + "=" * 70)
    print("CALL CENTER ASR BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Requests:     {total.num_requests} ({total.num_failures} failed)")
    print(f"Failure rate: {(total.num_failures / total.num_requests * 100) if total.num_requests else 0:.1f}%")
    print(f"RPS:          {total.total_rps:.2f}")
    print(f"Median:       {total.median_response_time:.0f}ms")
    print(f"P95:          {total.get_response_time_percentile(0.95):.0f}ms")
    print(f"P99:          {total.get_response_time_percentile(0.99):.0f}ms")

    if RTF_SAMPLES:
        rtf_arr = np.array(RTF_SAMPLES)
        lat_arr = np.array(LATENCY_SAMPLES)
        dur_arr = np.array(AUDIO_DURATIONS)

        print("-" * 70)
        print("RTF (Real-Time Factor — <1.0 = faster than real-time):")
        print(f"  Mean:   {np.mean(rtf_arr):.4f}")
        print(f"  Median: {np.median(rtf_arr):.4f}")
        print(f"  P95:    {np.percentile(rtf_arr, 95):.4f}")
        print(f"  P99:    {np.percentile(rtf_arr, 99):.4f}")
        print("-" * 70)
        print("Latency (ms):")
        print(f"  Mean:   {np.mean(lat_arr):.0f}")
        print(f"  Median: {np.median(lat_arr):.0f}")
        print(f"  P95:    {np.percentile(lat_arr, 95):.0f}")
        print(f"  P99:    {np.percentile(lat_arr, 99):.0f}")
        print("-" * 70)
        print(f"Audio processed: {np.sum(dur_arr):.0f}s ({np.sum(dur_arr)/3600:.2f} hours)")
        print(f"Avg audio duration: {np.mean(dur_arr):.2f}s")
        throughput_audio = np.sum(dur_arr) / (total.last_request_timestamp - total.start_time) if total.last_request_timestamp else 0
        print(f"Audio throughput: {throughput_audio:.1f}x real-time")

    if TEXTS:
        print("-" * 70)
        print("Sample transcriptions (first 5):")
        for i, t in enumerate(TEXTS[:5]):
            print(f"  [{i+1}] {t}")

    print("=" * 70)

    # Save JSON summary
    summary = {
        "benchmark_type": "callcenter_asr",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "source_lang": SOURCE_LANG,
            "target_lang": TARGET_LANG,
            "num_samples": len(_AUDIO_SAMPLES),
            "sample_durations": [d for _, _, d in _AUDIO_SAMPLES],
        },
        "total_requests": total.num_requests,
        "failures": total.num_failures,
        "rps": round(total.total_rps, 2),
        "median_ms": round(total.median_response_time, 0),
        "p95_ms": round(total.get_response_time_percentile(0.95), 0),
        "p99_ms": round(total.get_response_time_percentile(0.99), 0),
    }

    if RTF_SAMPLES:
        rtf_arr = np.array(RTF_SAMPLES)
        lat_arr = np.array(LATENCY_SAMPLES)
        dur_arr = np.array(AUDIO_DURATIONS)
        summary["rtf_mean"] = round(float(np.mean(rtf_arr)), 4)
        summary["rtf_median"] = round(float(np.median(rtf_arr)), 4)
        summary["rtf_p95"] = round(float(np.percentile(rtf_arr, 95)), 4)
        summary["rtf_p99"] = round(float(np.percentile(rtf_arr, 99)), 4)
        summary["total_audio_s"] = round(float(np.sum(dur_arr)), 1)
        summary["latency_mean_ms"] = round(float(np.mean(lat_arr)), 0)
        summary["latency_median_ms"] = round(float(np.median(lat_arr)), 0)
        summary["latency_p95_ms"] = round(float(np.percentile(lat_arr, 95)), 0)

    summary_path = RESULTS_DIR / "asr_callcenter_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


# ---------------------------------------------------------------------------
# User profiles
# ---------------------------------------------------------------------------

class CallerBase(HttpUser):
    """Base class: each caller sends audio for transcription."""
    abstract = True

    def _transcribe(self, profile_name):
        filename, audio_bytes, audio_duration = random.choice(_AUDIO_SAMPLES)

        t0 = time.perf_counter()
        try:
            with self.client.post(
                f"/transcribe/raw?source_lang={SOURCE_LANG}&target_lang={TARGET_LANG}",
                data=audio_bytes,
                headers={"Content-Type": "audio/wav"},
                catch_response=True,
                name=f"/transcribe/raw [{profile_name}]",
            ) as response:
                latency_s = time.perf_counter() - t0
                latency_ms = latency_s * 1000

                if response.status_code == 200:
                    data = response.json()
                    text = data.get("text", "")

                    if not text.strip():
                        response.failure("Empty transcription")
                        return

                    response.success()

                    rtf = latency_s / audio_duration if audio_duration > 0 else 0

                    with _lock:
                        RTF_SAMPLES.append(rtf)
                        LATENCY_SAMPLES.append(latency_ms)
                        AUDIO_DURATIONS.append(audio_duration)
                        if len(TEXTS) < 20:
                            TEXTS.append(f"{filename}: {text}")
                else:
                    response.failure(f"HTTP {response.status_code}")
        except Exception:
            pass


class CallerRealistic(CallerBase):
    """Typical conversation turn: user speaks → ASR → system responds → user thinks."""
    weight = 6
    wait_time = between(5, 12)

    @task
    def speak(self):
        self._transcribe("realistic")


class CallerFast(CallerBase):
    """Quick confirmations and IVR navigation."""
    weight = 3
    wait_time = between(2, 5)

    @task
    def speak(self):
        self._transcribe("fast")


class CallerBurst(CallerBase):
    """Rapid-fire speech: dictation, reading lists."""
    weight = 1
    wait_time = between(0.5, 1.5)

    @task
    def speak(self):
        self._transcribe("burst")
