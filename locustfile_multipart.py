"""Locust: /transcribe multipart only, timestamps OFF."""
import os, random
from locust import HttpUser, between, task

SAMPLES_DIR = "/teamspace/studios/this_studio/samples"
_AUDIO = []
for f in sorted(os.listdir(SAMPLES_DIR)):
    if f.endswith((".wav",)):
        with open(os.path.join(SAMPLES_DIR, f), "rb") as fh:
            _AUDIO.append((f, fh.read()))

class MultipartUser(HttpUser):
    wait_time = between(0, 0)
    @task
    def transcribe(self):
        name, audio = random.choice(_AUDIO)
        self.client.post(
            "/transcribe",
            files={"audio": (name, audio, "audio/wav")},
            data={"source_lang": "en", "target_lang": "en", "timestamps": "false"},
            name="/transcribe",
        )
