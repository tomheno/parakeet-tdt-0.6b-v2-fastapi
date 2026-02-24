"""Qwen3-ASR engine — thin wrapper around vLLM AsyncLLM.

Adapted from Qwen3-ASR/runtime/vllm/asr_server.py.
"""

import io
import logging
import subprocess
import uuid
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger("asr_service.qwen3")


def decode_audio(data: bytes, sr: int = 16000) -> np.ndarray:
    """Decode audio to float32 PCM."""
    try:
        with io.BytesIO(data) as f:
            wav, orig_sr = sf.read(f, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if orig_sr != sr:
                indices = np.linspace(0, len(wav) - 1, int(len(wav) * sr / orig_sr))
                wav = np.interp(indices, np.arange(len(wav)), wav)
            return wav.astype(np.float32)
    except Exception:
        proc = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
            input=data, capture_output=True, timeout=60,
        )
        if proc.returncode != 0:
            raise ValueError("Audio decode failed")
        return np.frombuffer(proc.stdout, dtype=np.float32)


def parse_output(raw: str, forced_lang: Optional[str] = None) -> tuple[str, str]:
    """Parse model output -> (language, text)."""
    text = raw.strip()
    lang = forced_lang or ""

    if not forced_lang and text.startswith("language "):
        rest = text[9:]
        if "<asr_text>" in rest:
            parts = rest.split("<asr_text>", 1)
            lang = parts[0].strip()
            text = parts[1].replace("</asr_text>", "").strip()

    for tok in ["<asr_text>", "</asr_text>", "<|endoftext|>", "<|im_end|>"]:
        text = text.replace(tok, "")

    return lang.strip(), text.strip()


class ASREngine:
    """vLLM-based ASR engine for Qwen3-ASR models."""

    def __init__(
        self,
        model: str,
        gpu_memory: float = 0.85,
        max_seqs: int = 64,
        max_batched_tokens: int = 220000,
        enable_prefix_caching: bool = True,
        mm_processor_cache_gb: float = 4.0,
    ):
        self.model = model
        self.gpu_memory = gpu_memory
        self.max_seqs = max_seqs
        self.max_batched_tokens = max_batched_tokens
        self.enable_prefix_caching = enable_prefix_caching
        self.mm_processor_cache_gb = mm_processor_cache_gb
        self.engine = None
        self.processor = None
        self.sampling_params = None

    async def start(self):
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
        from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
        from vllm import ModelRegistry

        ModelRegistry.register_model(
            "Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration
        )

        logger.info(
            "Loading %s (max_seqs=%d, max_batched_tokens=%d, prefix_caching=%s, mm_cache_gb=%.1f)",
            self.model, self.max_seqs, self.max_batched_tokens,
            self.enable_prefix_caching, self.mm_processor_cache_gb,
        )

        engine_args = AsyncEngineArgs(
            model=self.model,
            gpu_memory_utilization=self.gpu_memory,
            max_num_seqs=self.max_seqs,
            max_num_batched_tokens=self.max_batched_tokens,
            dtype="bfloat16",
            trust_remote_code=True,
            limit_mm_per_prompt={"audio": 1},
            enable_prefix_caching=self.enable_prefix_caching,
            mm_processor_cache_gb=self.mm_processor_cache_gb,
        )

        vllm_config = engine_args.create_engine_config()
        self.engine = AsyncLLM.from_vllm_config(vllm_config)
        self.processor = Qwen3ASRProcessor.from_pretrained(self.model, fix_mistral_regex=True)
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=500)

        logger.info("Qwen3-ASR ready")

    def build_prompt(self, language: Optional[str] = None, context: str = "") -> str:
        """Build prompt with optional language forcing."""
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if language:
            prompt += f"language {language}<asr_text>"
        return prompt

    async def transcribe(
        self, audio: np.ndarray, language: Optional[str] = None, context: str = ""
    ) -> dict:
        request_id = str(uuid.uuid4())
        prompt = self.build_prompt(language, context)
        inputs = {"prompt": prompt, "multi_modal_data": {"audio": [audio]}}

        text = ""
        async for out in self.engine.generate(inputs, self.sampling_params, request_id):
            if out.finished:
                text = out.outputs[0].text

        lang, transcription = parse_output(text, language)
        return {"text": transcription, "language": lang, "duration": len(audio) / 16000.0}

    async def transcribe_stream(
        self, audio: np.ndarray, language: Optional[str] = None, context: str = ""
    ):
        request_id = str(uuid.uuid4())
        prompt = self.build_prompt(language, context)
        inputs = {"prompt": prompt, "multi_modal_data": {"audio": [audio]}}

        prev = 0
        async for out in self.engine.generate(inputs, self.sampling_params, request_id):
            text = out.outputs[0].text
            delta = text[prev:]
            prev = len(text)
            yield {"delta": delta, "finished": out.finished}
