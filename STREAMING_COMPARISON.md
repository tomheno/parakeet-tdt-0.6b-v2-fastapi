# Streaming STT Implementation Comparison

## Two Available Endpoints

### 1. `/stream` - Simple Timeout-Based (Original)
**File:** `streaming_server.py`

**Approach:**
- Uses simple 1.5s silence timeout to detect utterance end
- Immediate streaming inference (no VAD overhead)
- Simplest implementation

**Latency:**
- ~100-200ms for interim transcripts
- 1.5s delay for final transcript (silence timeout)

**Pros:**
- Minimal overhead
- Simplest code
- No external dependencies

**Cons:**
- Inaccurate utterance boundaries (fixed timeout)
- May cut off slow speakers
- May wait too long for fast speakers

### 2. `/stream/vad` - VAD-Enhanced (Recommended)
**File:** `streaming_server_vad.py`

**Approach:**
- Silero VAD detects natural speech boundaries
- Still uses streaming inference (no batching)
- Adaptive utterance detection

**Latency:**
- ~100-200ms for interim transcripts (same as above)
- 250ms delay for final transcript (configurable `min_silence_ms`)
- +32ms VAD processing per window

**Pros:**
- Accurate speech/silence detection
- Adapts to speaking patterns
- Better user experience (natural pauses)

**Cons:**
- Slightly higher CPU usage (VAD model)
- Small additional latency (~32ms per window)
- Downloads VAD model on first use (~2MB)

## Batching Approach (Not Implemented for Streaming)

**Your Code:** `batch_worker.py`

**Why not used for streaming:**
```
Audio → Queue → Wait (15ms) → Batch (max 4) → ASR
                   ^^^^^^
                Adds latency!
```

**Best use case:**
- **File-based transcription** (not real-time streaming)
- Processing multiple audio files efficiently
- When throughput > latency

**For streaming, we want:**
- Immediate processing (no wait)
- Per-utterance transcription (no batching)

## Latency Breakdown

```
Component                /stream    /stream/vad
────────────────────────────────────────────────
Audio capture            ~20ms      ~20ms
VAD processing           N/A        ~32ms
Streaming inference      ~100ms     ~100ms
Network (WebSocket)      ~10ms      ~10ms
────────────────────────────────────────────────
INTERIM transcript       ~130ms     ~162ms
FINAL transcript         1500ms     250ms
```

**Winner for low latency:** `/stream/vad` (250ms vs 1500ms for finals)

## Recommendations

### For LiveKit Integration
Use `/stream/vad` because:
1. Natural utterance boundaries → better UX
2. Faster final transcripts (250ms vs 1500ms)
3. Small latency increase (~32ms) is worth accuracy gain

### For Ultra-Low Latency (< 150ms)
Use `/stream` if:
- You don't care about accurate utterance boundaries
- You only need interim transcripts (no finals)
- Every millisecond counts

### For Batch Transcription (Files)
Use your `batch_worker.py` approach:
- Multiple audio files at once
- Higher throughput
- Latency doesn't matter

## Testing

### Start the server:
```bash
cd /workspace/repo
uv run uvicorn parakeet_service.main:app --host 0.0.0.0 --port 8000
```

### Test both endpoints:
```python
# Original (timeout-based)
ws = await websockets.connect("ws://localhost:8000/stream")

# VAD-enhanced (recommended)
ws = await websockets.connect("ws://localhost:8000/stream/vad")
```

## Configuration

### VAD Parameters (`streaming_server_vad.py:76-80`)
```python
session = StreamingSessionVAD(
    model,
    session_id,
    vad_threshold=0.6,      # Higher = stricter (0.0-1.0)
    min_silence_ms=250,     # Shorter = faster finals
    speech_pad_ms=120,      # Context before/after speech
)
```

**Tuning:**
- **Lower latency:** Decrease `min_silence_ms` (e.g., 150ms)
- **Reduce false positives:** Increase `vad_threshold` (e.g., 0.7)
- **Smoother transitions:** Increase `speech_pad_ms` (e.g., 200ms)

## Implementation Details

### Both implementations use:
✓ NeMo `CacheAwareStreamingAudioBuffer` for streaming inference
✓ Same encoder/decoder pipeline (no `transcribe()` API issues)
✓ WebSocket protocol for real-time communication
✓ Session management for multiple concurrent clients

### Key difference:
- `/stream`: Uses time-based silence detection
- `/stream/vad`: Uses ML-based speech detection (Silero VAD)