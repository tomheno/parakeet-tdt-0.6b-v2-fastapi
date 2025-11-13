# Parakeet TDT Streaming STT for LiveKit

Low-latency (<300ms) streaming speech-to-text using NVIDIA's Parakeet-TDT model, optimized for LiveKit agents.

## Features

- **True Streaming Inference** - Cache-aware processing with NeMo
- **Low Latency** - <300ms end-to-end with 80ms chunks
- **LiveKit Integration** - Full STT plugin for LiveKit agents
- **Multi-language** - 25+ languages with auto-detection
- **GPU Optimized** - Efficient batch processing on NVIDIA GPUs
- **Production Ready** - Docker, dstack, auto-scaling support

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/tomheno/parakeet-tdt-0.6b-v2-fastapi.git
cd parakeet-tdt-0.6b-v2-fastapi

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Or with LiveKit support
uv sync --extra livekit
```

### Run Server

```bash
# Local development
uv run uvicorn parakeet_service.main:app --host 0.0.0.0 --port 8000

# With GPU
MODEL_PRECISION=fp16 DEVICE=cuda uv run uvicorn parakeet_service.main:app
```

### Test

```bash
# Install test dependencies
uv sync --extra test

# Sanity tests (no server needed)
uv run pytest -m sanity

# Integration tests (requires running server)
uv run pytest -m integration

# Test with microphone
uv run python test_microphone.py
```

## LiveKit Integration

### Plugin Usage

```python
from livekit_plugin import ParakeetSTT

# Initialize STT
stt = ParakeetSTT(
    model_name="nvidia/parakeet-tdt-0.6b-v3",
    chunk_size=0.08,  # 80ms
    precision="fp16"
)

# Use in LiveKit agent
async with stt.stream() as stream:
    async for event in stream:
        if event.type == SpeechEventType.INTERIM_TRANSCRIPT:
            print(event.alternatives[0].text)
```

### LiveKit Agent Example

```python
from livekit.agents import JobContext, cli
from livekit_plugin import ParakeetSTT

async def entrypoint(ctx: JobContext):
    stt = ParakeetSTT()

    @ctx.on("track_subscribed")
    async def on_track(track):
        if track.kind == "audio":
            async for event in stt.stream():
                print(f"Transcript: {event.alternatives[0].text}")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

## WebSocket API

### Endpoint: `ws://localhost:8000/stream`

**Client sends:**
- PCM int16 audio bytes
- Mono, 16kHz
- 80ms chunks (1280 samples)

**Server responds:**
```json
{
  "text": "transcribed text",
  "is_final": false,
  "session_id": "uuid"
}
```

### Example Client

```python
import asyncio
import websockets
import numpy as np

async def stream_audio():
    async with websockets.connect("ws://localhost:8000/stream") as ws:
        # Send 80ms audio chunks
        audio = np.random.randn(1280).astype(np.float32) * 0.01
        audio_int16 = (audio * 32768).astype(np.int16)

        await ws.send(audio_int16.tobytes())

        # Receive transcript
        response = await ws.recv()
        print(response)

asyncio.run(stream_audio())
```

## Deployment

### Docker

```bash
# Build
docker build -f Dockerfile.gpu -t parakeet-stt:gpu .

# Run
docker run --gpus all -p 8000:8000 parakeet-stt:gpu
```

### Docker Compose

```bash
docker-compose -f docker-compose.gpu.yml up
```

### dstack (Multi-cloud)

```bash
uv tool install dstack
dstack init
dstack run .
```

Configuration in `.dstack.yml`:
```yaml
type: service
image: ${{ run.dockerfile }}
dockerfile: Dockerfile.gpu

resources:
  gpu:
    memory: 16GB..24GB
    name: A10G

ports:
  - 8000
```

## Performance

### Latency Targets

| Configuration | Latency | Accuracy | Use Case |
|---------------|---------|----------|----------|
| Aggressive (2 chunks) | 150ms | Good | Real-time chat |
| Balanced (4 chunks) | 250ms | High | LiveKit agents |
| Conservative (8 chunks) | 400ms | Highest | High accuracy |

### GPU Requirements

| GPU | VRAM | Concurrent Streams | Cost/hr |
|-----|------|-------------------|---------|
| T4 | 16GB | ~10 | $0.35 |
| A10G | 24GB | ~20 | $1.00 |
| A100 | 40GB | ~40 | $3.00 |

### Throughput

- Single stream: ~1.0x realtime (80ms processing per 80ms audio)
- Concurrent streams: 20+ on A10G GPU
- Cost: ~$0.0003/minute (vs Deepgram $0.0009/min)

## Configuration

### Environment Variables

```bash
MODEL_PRECISION=fp16     # fp16 or fp32
DEVICE=cuda              # cuda or cpu
LOG_LEVEL=INFO           # DEBUG, INFO, WARNING, ERROR
```

### Model Selection

```bash
# Default (v3)
MODEL_NAME=nvidia/parakeet-tdt-0.6b-v3

# Alternative (v2)
MODEL_NAME=nvidia/parakeet-tdt-0.6b-v2
```

## Testing

### Run All Tests

```bash
# Install test dependencies
uv sync --extra test

# Run all tests
uv run pytest
```

### Test Categories

```bash
# Sanity tests (fast, no server)
uv run pytest -m sanity

# Integration tests (requires server)
uv run pytest -m integration

# Remote tests (requires dstack deployment)
uv run pytest -m remote
```

### Interactive Testing

```bash
# Test with audio file
uv run python test_streaming_client.py audio.wav

# Test with microphone (local)
uv run python test_microphone.py

# Test with microphone (dstack)
uv run python test_microphone.py --remote

# Test custom endpoint
uv run python test_microphone.py --url wss://your-server.com
```

## Architecture

```
┌─────────────┐
│   Client    │
│ (LiveKit)   │
└──────┬──────┘
       │ WebSocket (80ms PCM chunks)
       ↓
┌─────────────────────────────────┐
│   FastAPI Server                │
│   - /healthz (health check)     │
│   - /stream (WebSocket STT)     │
└──────┬──────────────────────────┘
       │
       ↓
┌─────────────────────────────────┐
│   StreamingSession              │
│   - CacheAwareStreamingBuffer   │
│   - 80ms chunks                 │
│   - 4 chunk context (320ms)     │
└──────┬──────────────────────────┘
       │
       ↓
┌─────────────────────────────────┐
│   Parakeet TDT Model            │
│   - NeMo ASR                    │
│   - FP16 precision              │
│   - Cache-aware attention       │
└─────────────────────────────────┘
```

## Documentation

- [STREAMING.md](STREAMING.md) - Detailed streaming guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Multi-cloud deployment
- [TESTING.md](TESTING.md) - Complete testing guide

## Supported Languages

Parakeet TDT supports 25+ languages with auto-detection:

- English (en), Spanish (es), French (fr), German (de)
- Italian (it), Portuguese (pt), Polish (pl), Turkish (tr)
- Russian (ru), Dutch (nl), Czech (cs), Arabic (ar)
- Chinese (zh), Japanese (ja), Korean (ko), Hindi (hi)
- Persian (fa), Ukrainian (uk), Vietnamese (vi), and more

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.1+ (recommended)
- 16GB+ GPU VRAM for production
- NeMo Toolkit
- FastAPI, WebSockets
- LiveKit Agents SDK (optional)

## Troubleshooting

### Import Errors

```bash
uv pip install --upgrade nemo_toolkit[asr]
```

### GPU Not Detected

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### No Transcripts Received

- Model needs real speech (test audio/noise may not produce transcripts)
- Check audio format (must be 16kHz mono)
- Increase chunk context: `left_chunks=8`

### High Latency

- Reduce chunk context: `left_chunks=2`
- Use FP16: `MODEL_PRECISION=fp16`
- Check GPU utilization: `nvidia-smi`

## License

MIT

## Acknowledgments

- NVIDIA NeMo Team for Parakeet TDT model
- LiveKit for real-time communication platform
- dstack for multi-cloud GPU deployment
