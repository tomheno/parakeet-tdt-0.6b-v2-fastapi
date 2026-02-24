# ASR Benchmark Methodology

Standard methodology for comparing ASR backends (Canary vs Qwen3) on identical hardware.

## Hardware

- **GPU**: NVIDIA H100 80GB HBM3
- **CPU**: As provided by Lightning Studios
- **Driver/CUDA**: As installed in conda environment

## Audio Corpus

6 real speech WAV files from `/teamspace/studios/this_studio/samples/`, 16kHz mono:

| File | Duration |
|------|----------|
| angry_fast.wav | 4.04s |
| casual_indecise.wav | 1.89s |
| neutral_fast.wav | 2.35s |
| polite_slow.wav | 5.71s |
| short_us_f.wav | 2.08s |
| short_us_f2.wav | 1.94s |
| **Average** | **3.00s** |

Each request uses a randomly selected sample.

## Server Configuration

### Canary (nvidia/canary-1b-v2)
```bash
ASR_BACKEND=canary OPT_TF32=1 OPT_GREEDY=1 OPT_SDPA=1 OPT_SELF_KV=1 NO_TIMESTAMPS=1 \
  python -c "import uvicorn; uvicorn.run('asr_service.main:app', host='0.0.0.0', port=8000)"
```
- FP16, greedy decoding, SDPA + KV caches, TF32, no timestamps
- Single uvicorn worker

### Qwen3 (Qwen/Qwen3-ASR-0.6B or 1.7B)
```bash
ASR_BACKEND=qwen3 QWEN3_MODEL=Qwen/Qwen3-ASR-0.6B \
  python -c "import uvicorn; uvicorn.run('asr_service.main:app', host='0.0.0.0', port=8000)"
```
- BF16, vLLM defaults
- Single uvicorn worker

## Benchmark Methods

### 1. Async Throughput Sweep (Primary — True Server Ceiling)

Uses aiohttp async client to bypass locust's connection pool bottleneck.
30 seconds per concurrency level. Concurrency levels: 16, 32, 64, 128, 256, 512.

```bash
python benchmark/async_sweep.py --host http://localhost:8000 --duration 30
```

Or inline:
```python
# See plan file for full async sweep script
```

**Endpoint**: `POST /transcribe/raw` (for Canary native) or `POST /v1/audio/transcriptions` (unified)

### 2. Locust Load Test (Secondary — Standardized)

```bash
for u in 16 32 64 128 256 512; do
    locust -f benchmark/locustfile_openai.py --host http://localhost:8000 \
        --headless -u $u -r $u --run-time 30s --csv results/openai_u${u}
done
```

**Endpoint**: `POST /v1/audio/transcriptions` (multipart form — higher overhead than raw)

### 3. Call Center Realistic (Qualitative)

```bash
locust -f locustfile_callcenter.py --host http://localhost:8000 \
    --headless -u 1000 -r 50 --run-time 90s
```

## Metrics Collected

| Metric | Definition |
|--------|-----------|
| **RPS** | Successful requests per second |
| **P50/P95/P99** | Latency percentiles in milliseconds |
| **Per-request RTFX** | `audio_duration / latency` — higher is better (>1.0 = faster than real-time) |
| **Aggregate RTFX** | `total_audio_seconds / wall_time` — throughput in audio-seconds-per-second |
| **Error rate** | HTTP errors or empty transcriptions |

## Apple-to-Apple Comparison Rules

1. **Same audio corpus** — identical 6 WAV files
2. **Same endpoint** — `POST /v1/audio/transcriptions` for unified comparison
3. **Same client** — aiohttp async sweep with identical parameters
4. **Same hardware** — same GPU, same machine
5. **Same concurrency levels** — 16, 32, 64, 128, 256, 512
6. **Same duration** — 30 seconds per concurrency level
7. **Warm start** — discard first 5 seconds of results (warmup)
8. **Report all** — RPS, P50/P95/P99, RTFX, errors

## Canary Baseline Results (Pre-Merge)

Config: `OPT_TF32=1 OPT_GREEDY=1 OPT_SDPA=1 OPT_SELF_KV=1 NO_TIMESTAMPS=1`
Endpoint: `POST /transcribe/raw` (native, lowest overhead)

| Concurrency | RPS | P50 | P95 | P99 | Per-req RTFX | Agg RTFX | Errors |
|-------------|-----|-----|-----|-----|--------------|----------|--------|
| 16 | 71 | 425ms | 462ms | 538ms | 5.9x | 216x | 0 |
| 32 | 133 | 438ms | 469ms | 498ms | 5.4x | 400x | 0 |
| 64 | 191 | 486ms | 536ms | 629ms | 4.9x | 575x | 0 |
| 128 | 288 | 584ms | 837ms | 1,310ms | 3.9x | 859x | 0 |
| 256 | 533 | 758ms | 1,026ms | 1,053ms | 3.0x | 1,588x | 0 |
| 512 | 697 | 1,065ms | 1,353ms | 1,741ms | 2.0x | 2,107x | 0 |

**Peak: 697 RPS @ 512 concurrency, zero errors.**
