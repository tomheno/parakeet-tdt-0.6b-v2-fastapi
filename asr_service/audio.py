"""Shared audio decoding utilities."""

import io
import subprocess

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def decode_audio(data: bytes, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decode audio bytes to float32 mono numpy at target_sr.

    Uses soundfile with ffmpeg fallback for broad format support.
    """
    try:
        with io.BytesIO(data) as f:
            wav, orig_sr = sf.read(f, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if orig_sr != target_sr:
                indices = np.linspace(0, len(wav) - 1, int(len(wav) * target_sr / orig_sr))
                wav = np.interp(indices, np.arange(len(wav)), wav)
            return wav.astype(np.float32)
    except Exception:
        proc = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "pipe:1"],
            input=data, capture_output=True, timeout=60,
        )
        if proc.returncode != 0:
            raise ValueError("Audio decode failed (soundfile + ffmpeg both failed)")
        return np.frombuffer(proc.stdout, dtype=np.float32)
