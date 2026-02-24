import os

# Model
QWEN3_MODEL = os.getenv("QWEN3_MODEL", "Qwen/Qwen3-ASR-1.7B")

# vLLM engine params
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85"))
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "256"))
MAX_BATCHED_TOKENS = int(os.getenv("MAX_BATCHED_TOKENS", "220000"))
ENABLE_PREFIX_CACHING = os.getenv("ENABLE_PREFIX_CACHING", "1").lower() in ("1", "true", "yes")
MM_PROCESSOR_CACHE_GB = float(os.getenv("MM_PROCESSOR_CACHE_GB", "4.0"))
