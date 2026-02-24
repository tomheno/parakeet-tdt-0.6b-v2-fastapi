"""Unified ASR benchmark — targets OpenAI-compatible /v1/audio/transcriptions.

Works with both Canary and Qwen3 backends. Designed for apple-to-apple comparison.

Usage:
    locust -f benchmark/locustfile_openai.py --host http://localhost:8000 \
        --headless -u 100 -r 50 --run-time 60s

    # Sweep to find saturation point:
    for u in 16 32 64 128 256 512; do
        locust -f benchmark/locustfile_openai.py --host http://localhost:8000 \
            --headless -u $u -r $u --run-time 30s \
            --csv results/openai_u${u}
    done
"""

import os
import random
import time

import soundfile as sf
from locust import HttpUser, between, events, task

SAMPLES_DIR = os.getenv("SAMPLES_DIR", "/teamspace/studios/this_studio/samples")

# Loaded at init
_samples = []  # [(filename, bytes, duration_s)]


@events.init.add_listener
def on_init(environment, **kwargs):
    global _samples
    for fname in sorted(os.listdir(SAMPLES_DIR)):
        if not fname.endswith(".wav"):
            continue
        path = os.path.join(SAMPLES_DIR, fname)
        data, sr = sf.read(path)
        duration = len(data) / sr
        with open(path, "rb") as f:
            audio_bytes = f.read()
        _samples.append((fname, audio_bytes, duration))
    if not _samples:
        raise RuntimeError(f"No WAV files found in {SAMPLES_DIR}")
    avg_dur = sum(d for _, _, d in _samples) / len(_samples)
    print(f"Loaded {len(_samples)} audio samples (avg {avg_dur:.2f}s) from {SAMPLES_DIR}")


class TranscribeUser(HttpUser):
    """Zero-think-time user — max throughput benchmark."""
    wait_time = between(0, 0)

    @task
    def transcribe(self):
        fname, audio_bytes, duration = random.choice(_samples)
        t0 = time.perf_counter()
        with self.client.post(
            "/v1/audio/transcriptions",
            files={"file": (fname, audio_bytes, "audio/wav")},
            data={"model": "auto", "response_format": "json"},
            catch_response=True,
            name="/v1/audio/transcriptions",
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if not body.get("text"):
                    resp.failure("Empty transcription")
            else:
                resp.failure(f"HTTP {resp.status_code}")
