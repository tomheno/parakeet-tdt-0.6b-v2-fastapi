"""
Locust load test for Canary 1B V2 service.

Usage:
    # Headless with CSV stats:
    locust --headless -u 64 -r 16 -t 60s --csv=results/h100 -H http://localhost:8000

    # Web UI:
    locust -H http://localhost:8000
"""

import os
import random

from locust import HttpUser, between, task

SAMPLES_DIR = "/teamspace/studios/this_studio/samples"

# Pre-load audio files once
_AUDIO_FILES = []
for f in sorted(os.listdir(SAMPLES_DIR)):
    if f.endswith((".wav", ".flac", ".mp3")):
        path = os.path.join(SAMPLES_DIR, f)
        with open(path, "rb") as fh:
            _AUDIO_FILES.append((f, fh.read()))


class TranscribeUser(HttpUser):
    """Simulates users sending transcription requests."""

    wait_time = between(0, 0)  # no wait — max throughput

    @task(3)
    def transcribe_multipart(self):
        """POST /transcribe with multipart form."""
        name, audio = random.choice(_AUDIO_FILES)
        self.client.post(
            "/transcribe",
            files={"audio": (name, audio, "audio/wav")},
            data={"source_lang": "en", "target_lang": "en", "timestamps": "false"},
            name="/transcribe",
        )

    @task(3)
    def transcribe_raw(self):
        """POST /transcribe/raw with binary body."""
        name, audio = random.choice(_AUDIO_FILES)
        self.client.post(
            "/transcribe/raw?source_lang=en&target_lang=en&timestamps=false",
            data=audio,
            headers={"Content-Type": "audio/wav"},
            name="/transcribe/raw",
        )

    @task(2)
    def transcribe_multipart_timestamps(self):
        """POST /transcribe with timestamps enabled."""
        name, audio = random.choice(_AUDIO_FILES)
        self.client.post(
            "/transcribe",
            files={"audio": (name, audio, "audio/wav")},
            data={"source_lang": "en", "target_lang": "en", "timestamps": "true"},
            name="/transcribe [ts]",
        )

    @task(2)
    def transcribe_raw_timestamps(self):
        """POST /transcribe/raw with timestamps enabled."""
        name, audio = random.choice(_AUDIO_FILES)
        self.client.post(
            "/transcribe/raw?source_lang=en&target_lang=en&timestamps=true",
            data=audio,
            headers={"Content-Type": "audio/wav"},
            name="/transcribe/raw [ts]",
        )
