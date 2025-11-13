# Parakeet TDT Streaming STT

Low-latency (<300ms) streaming speech-to-text using NVIDIA Parakeet TDT with cache-aware inference.

## Quick Start

### 1. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 2. Start Server

```bash
uvicorn parakeet_service.main:app --host 0.0.0.0 --port 8000
```

### 3. Test Streaming

```bash
python test_streaming_client.py your_audio.wav
```

## Architecture

### Streaming Modes

**Legacy (`/ws`)**: VAD-based batch processing
- High latency (~2s)
- Uses Silero VAD + queue + batch worker
- Good for utterance-based transcription

**New (`/stream`)**: Cache-aware streaming
- Low latency (<300ms)
- Direct inference with `CacheAwareStreamingAudioBuffer`
- Continuous interim transcripts
- **Recommended for LiveKit**

## WebSocket Protocol

### Endpoint: `ws://localhost:8000/stream`

**Client sends:**
```
Binary frames: PCM int16, mono, 16kHz
Chunk size: 1280 samples (80ms)
```

**Server responds:**
```json
{
  "text": "transcribed text",
  "is_final": false,
  "session_id": "uuid"
}
```

## LiveKit Integration

### Installation

```bash
uv pip install livekit-agents
```

### Usage

```python
from livekit_plugin import ParakeetSTT

# Initialize
stt = ParakeetSTT(
    model_name="nvidia/parakeet-tdt-0.6b-v3",
    chunk_size=0.08,  # 80ms
    precision="fp16"
)

# Stream audio
async with stt.stream() as stream:
    # Feed audio frames
    stream.push_frame(audio_frame)

    # Receive transcripts
    async for event in stream:
        if event.type == SpeechEventType.INTERIM_TRANSCRIPT:
            print(event.alternatives[0].text)
```

### LiveKit Agent Example

```python
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit_plugin import ParakeetSTT

async def entrypoint(ctx: JobContext):
    stt = ParakeetSTT()

    @ctx.on("track_subscribed")
    async def on_track_subscribed(track: rtc.Track):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_stream = rtc.AudioStream(track)

            stt_stream = stt.stream()
            async for event in stt_stream:
                print(f"Transcript: {event.alternatives[0].text}")

                # Feed audio to STT
                async for frame in audio_stream:
                    stt_stream.push_frame(frame)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

## Performance Tuning

### Latency vs Accuracy

```python
# Lowest latency (150ms) - reduced accuracy
CacheAwareStreamingAudioBuffer(
    chunk_size=0.08,
    shift_size=0.04,
    left_chunks=2    # Less context
)

# Balanced (250ms) - recommended
CacheAwareStreamingAudioBuffer(
    chunk_size=0.08,
    shift_size=0.04,
    left_chunks=4    # 320ms context
)

# Higher accuracy (400ms) - more latency
CacheAwareStreamingAudioBuffer(
    chunk_size=0.16,
    shift_size=0.08,
    left_chunks=8    # 1.28s context
)
```

### GPU Memory

- Model: ~1.2GB (FP16)
- Per stream cache: ~100MB
- **Max concurrent streams**: ~20 on 24GB GPU

## Troubleshooting

### CacheAwareStreamingAudioBuffer not found

If NeMo's streaming utils aren't available, the code falls back to chunked inference:

```bash
# Update NeMo to latest
uv pip install --upgrade nemo_toolkit[asr]
```

### High latency

1. Check chunk size (should be 80ms)
2. Reduce `left_chunks` (less context = lower latency)
3. Verify network RTT with `ping`
4. Check GPU utilization

### Import errors

```bash
# Install all dependencies
uv pip install fastapi uvicorn websockets soundfile
uv pip install nemo_toolkit[asr]
uv pip install livekit-agents
```

## Files

```
parakeet-tdt-0.6b-v2-fastapi/
├── parakeet_service/
│   ├── streaming_server.py      # Core streaming logic
│   └── main.py                   # FastAPI app with /stream endpoint
├── livekit_plugin.py             # LiveKit STT plugin
├── test_streaming_client.py      # WebSocket test client
└── STREAMING.md                  # This file
```

## API Comparison

| Feature | Legacy `/ws` | Streaming `/stream` |
|---------|--------------|---------------------|
| Latency | ~2000ms | <300ms |
| Processing | Batch | Real-time |
| VAD | Yes (Silero) | No |
| Interim results | Limited | Continuous |
| Memory/stream | Low | ~100MB |
| Accuracy | High | High |
| Use case | Utterances | Live conversation |

## References

- NeMo Streaming: `nemo/collections/asr/parts/utils/streaming_utils.py`
- Parakeet Model: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- LiveKit Agents: https://docs.livekit.io/agents/
