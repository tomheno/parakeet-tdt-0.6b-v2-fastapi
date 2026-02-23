"""Locust: /transcribe/raw only, timestamps OFF — max throughput test."""
import os, random
from locust import HttpUser, between, task

SAMPLES_DIR = "/teamspace/studios/this_studio/samples"
_AUDIO = []
for f in sorted(os.listdir(SAMPLES_DIR)):
    if f.endswith((".wav",)):
        with open(os.path.join(SAMPLES_DIR, f), "rb") as fh:
            _AUDIO.append(fh.read())

class RawUser(HttpUser):
    wait_time = between(0, 0)
    @task
    def raw_off(self):
        audio = random.choice(_AUDIO)
        self.client.post(
            "/transcribe/raw?source_lang=en&target_lang=en&timestamps=false",
            data=audio, headers={"Content-Type": "audio/wav"},
            name="/transcribe/raw",
        )
