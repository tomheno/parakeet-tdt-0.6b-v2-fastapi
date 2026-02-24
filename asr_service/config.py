import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Backend selection: "canary" or "qwen3"
ASR_BACKEND = os.getenv("ASR_BACKEND", "canary").lower()

# Audio
TARGET_SR = 16000

# Server
PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", "120"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger("asr_service")
