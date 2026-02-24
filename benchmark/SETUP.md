# Setup & Run Benchmarks on New Hardware

End-to-end guide: bare metal or cloud VM with NVIDIA GPU → running server → benchmark results.

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | NVIDIA V100 (SM 70, 32 GB) | H100 80 GB HBM3 |
| CUDA | 12.1 | 12.1+ |
| Python | 3.10 | 3.10–3.12 |
| RAM | 8 GB | 16 GB+ |
| Disk | 10 GB (model cache) | 20 GB |

## 1. System Dependencies

```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential ffmpeg libsndfile1 git curl
```

`ffmpeg` and `libsndfile1` are **required** — audio decoding will fail without them.

## 2. Python Environment

```bash
# Option A: UV (fast, recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras          # installs torch with CUDA 12.1 automatically

# Option B: conda/pip
pip install -e ".[test]"
pip install locust aiohttp
```

Verify CUDA:
```bash
python -c "import torch; print(torch.cuda.get_device_name())"
```

## 3. Compile Fast Copy Extension (Optional)

GIL-free batch padding — runs audio copy in parallel with GPU. Falls back to numpy if missing.

```bash
cd asr_service/backends/canary
gcc -O3 -march=native -shared -fPIC -o _fast_copy.so _fast_copy.c
```

## 4. Audio Samples

Benchmarks expect 6 WAV files in `samples/` (16 kHz mono, ~3s average):

```
samples/
├── angry_fast.wav        (4.04s)
├── casual_indecise.wav   (1.89s)
├── neutral_fast.wav      (2.35s)
├── polite_slow.wav       (5.71s)
├── short_us_f.wav        (2.08s)
└── short_us_f2.wav       (1.94s)
```

If you need different samples, place WAV files there. Update `SAMPLES_DIR` env var in benchmark scripts if using a different path.

## 5. Start the Server

```bash
OPT_TF32=1 OPT_GREEDY=1 OPT_SDPA=1 OPT_SELF_KV=1 NO_TIMESTAMPS=1 \
  python -c "import uvicorn; uvicorn.run('asr_service.main:app', host='0.0.0.0', port=8000)"
```

First start downloads `nvidia/canary-1b-v2` (~4 GB) from HuggingFace and warms up CUDA graphs. Takes 60–90s. Wait for:

```
INFO: Canary 1B V2 ready on cuda:0
INFO: Uvicorn running on http://0.0.0.0:8000
```

Verify:
```bash
curl http://localhost:8000/health
# → {"status":"ok","backend":"canary"}

curl -X POST http://localhost:8000/transcribe/raw \
  --data-binary @samples/short_us_f.wav \
  -H "Content-Type: audio/wav"
# → {"text":"Can you help me with my order please?"}
```

### Environment Variables Reference

**Optimization flags** (all default ON except where noted):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPT_TF32` | `1` | TF32 matmul + cuDNN auto-tune |
| `OPT_GREEDY` | `1` | Greedy decoding (vs beam search) |
| `OPT_SDPA` | `1` | Scaled dot-product attention + cross-attn KV cache |
| `OPT_SELF_KV` | `1` | Self-attention KV buffer (greedy only) |
| `NO_TIMESTAMPS` | `0` | Drop CTC timestamps model (~200 MB VRAM savings) |
| `BEAM_SIZE` | `0` | Fixed beam mode (0 = dynamic per-request) |
| `MAX_GEN_DELTA` | `50` | Max decoder output tokens beyond encoder length |
| `OPT_COMPILE_DEC` | `0` | torch.compile decoder (off by default — regression) |

**Tuning knobs:**

| Variable | Default | Tune for |
|----------|---------|----------|
| `MAX_BATCH_SIZE` | `256` | Lower if OOM, raise if GPU under-utilized |
| `AUDIO_WORKERS` | `16` | CPU thread pool for audio decoding |

**Per-request beam search:**
```bash
# Default requests use greedy. Override per-request via query param:
curl "http://localhost:8000/transcribe/raw?beam_size=2" \
  --data-binary @audio.wav -H "Content-Type: audio/wav"
```

## 6. Run Benchmarks

### Quick Sanity Check (1 minute)

```bash
python3 -c "
import asyncio, aiohttp, time, os, numpy as np, soundfile as sf, io

async def quick():
    samples = []
    for f in sorted(os.listdir('samples')):
        if f.endswith('.wav'):
            data, sr = sf.read(f'samples/{f}')
            buf = io.BytesIO(); sf.write(buf, data, sr, format='WAV')
            samples.append(buf.getvalue())
    async with aiohttp.ClientSession() as s:
        for audio in samples:
            t0 = time.monotonic()
            async with s.post('http://localhost:8000/transcribe/raw',
                data=audio, headers={'Content-Type':'audio/wav'}) as r:
                result = await r.json()
                print(f'{(time.monotonic()-t0)*1000:.0f}ms  {result[\"text\"]}')
asyncio.run(quick())
"
```

### Full Concurrency Sweep (greedy, ~4 minutes)

```bash
python3 -c "
import asyncio, aiohttp, time, random, os, numpy as np, soundfile as sf, io

SAMPLES_DIR = 'samples'
HOST = 'http://localhost:8000'

async def bench(conc, duration=30):
    samples = []
    for f in sorted(os.listdir(SAMPLES_DIR)):
        if f.endswith('.wav'):
            data, sr = sf.read(f'{SAMPLES_DIR}/{f}')
            buf = io.BytesIO(); sf.write(buf, data, sr, format='WAV')
            samples.append(buf.getvalue())
    latencies, errors, stop = [], 0, False
    async def worker(session):
        nonlocal errors, stop
        while not stop:
            t0 = time.monotonic()
            try:
                async with session.post(f'{HOST}/transcribe/raw',
                    data=random.choice(samples),
                    headers={'Content-Type':'audio/wav'}) as r:
                    if r.status == 200:
                        await r.json(); latencies.append(time.monotonic() - t0)
                    else: errors += 1
            except: errors += 1
    conn = aiohttp.TCPConnector(limit=conc+100)
    async with aiohttp.ClientSession(connector=conn,
        timeout=aiohttp.ClientTimeout(total=120)) as session:
        # warmup
        async with session.post(f'{HOST}/transcribe/raw',
            data=samples[0], headers={'Content-Type':'audio/wav'}) as r:
            await r.json()
        tasks = [asyncio.create_task(worker(session)) for _ in range(conc)]
        await asyncio.sleep(duration); stop = True
        await asyncio.gather(*tasks, return_exceptions=True)
    arr = np.array(latencies)
    rps = len(latencies) / duration
    p50 = np.percentile(arr, 50)*1000
    p95 = np.percentile(arr, 95)*1000
    p99 = np.percentile(arr, 99)*1000
    cum_rtfx = rps * 3.0  # avg sample = 3.0s
    return conc, rps, p50, p95, p99, cum_rtfx, errors

async def main():
    gpu = 'unknown'
    try:
        import torch; gpu = torch.cuda.get_device_name()
    except: pass
    print(f'GPU: {gpu}')
    print(f'{\"conc\":>5} | {\"RPS\":>7} | {\"P50\":>8} | {\"P95\":>8} | {\"P99\":>8} | {\"cRTFx\":>7} | {\"err\":>4}')
    print('-' * 68)
    for conc in [16, 32, 64, 128, 256, 512]:
        c, rps, p50, p95, p99, crtfx, err = await bench(conc, 30)
        print(f'{c:>5} | {rps:>7.0f} | {p50:>7.0f}ms | {p95:>7.0f}ms | {p99:>7.0f}ms | {crtfx:>6.0f}x | {err:>4}')
        await asyncio.sleep(2)

asyncio.run(main())
"
```

### Locust (Alternative)

```bash
# Single concurrency level
locust -f benchmark/locustfile_openai.py --host http://localhost:8000 \
    --headless -u 256 -r 256 --run-time 30s --csv results/my_gpu_u256

# Full sweep
for u in 16 32 64 128 256 512; do
    locust -f benchmark/locustfile_openai.py --host http://localhost:8000 \
        --headless -u $u -r $u --run-time 30s --csv results/my_gpu_u${u}
done
```

Note: locust uses multipart form-data (higher overhead than raw binary). The async sweep above uses raw binary for true server ceiling measurement.

## 7. Save Results

Save to `results/greedy_<gpu>.csv`:

```csv
gpu,mode,concurrency,rps,p50_ms,p95_ms,p99_ms,rtfx_p50,cumulative_rtfx,errors,duration_s
L40S 46GB,greedy,16,36,349,1141,2099,8.6,108,0,30
...
```

Column definitions:
- **rtfx_p50**: Per-request real-time factor = `avg_audio_duration / (p50_ms / 1000)` — how many times faster than real-time a single request completes
- **cumulative_rtfx**: Aggregate throughput = `rps * avg_audio_duration` — total seconds of audio processed per second of wall clock

## Known Baselines

| GPU | Peak RPS | Peak cRTFx | Best Latency (P50) |
|-----|----------|-----------|-------------------|
| H100 80GB HBM3 | 521 @ 512u | 1,563x | 425ms @ 64u |
| L40S 46GB | 273 @ 512u | 819x | 349ms @ 16u |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `soundfile.LibsndfileError` | `sudo apt install libsndfile1` |
| `FileNotFoundError: ffmpeg` | `sudo apt install ffmpeg` |
| CUDA OOM at large batch | Set `MAX_BATCH_SIZE=128` (or lower) |
| Low RPS despite high concurrency | Check `nvidia-smi` — another process may hold the GPU |
| `lhotse` / `DynamicCutSampler` crash | Already monkey-patched in `model.py` for PyTorch >=2.10 |
| `_fast_copy.so` not found | Compile it (step 3) or ignore — numpy fallback works |
| First batch very slow | Normal — CUDA graph warmup on first shapes, stabilizes after ~5s |
