import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "nvidia/canary-1b-v2"

# Audio
TARGET_SR = 16000

# Supported languages (Canary 1B V2 — 25 European languages)
SUPPORTED_LANGUAGES = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
    "sl", "es", "sv", "ru", "uk",
]

# Inference
MODEL_PRECISION = os.getenv("MODEL_PRECISION", "fp16")
DEVICE = os.getenv("DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "60"))  # seconds

# Streaming buffer settings
TRANSCRIBE_INTERVAL_S = float(os.getenv("TRANSCRIBE_INTERVAL_S", "1.0"))
SILENCE_TIMEOUT_S = float(os.getenv("SILENCE_TIMEOUT_S", "1.5"))

# Server
PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", "120"))  # seconds
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger("canary_service")
