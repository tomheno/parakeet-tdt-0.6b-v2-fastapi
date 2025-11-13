#!/usr/bin/env python3
"""
Test client for Parakeet streaming STT WebSocket endpoint.

Usage:
    python test_streaming_client.py <audio_file.wav>

Requirements:
    pip install websockets soundfile numpy
"""
import asyncio
import websockets
import numpy as np
import time
import sys
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    print("Error: soundfile not installed. Run: pip install soundfile")
    sys.exit(1)


async def stream_audio_file(file_path: str, ws_url: str = "ws://localhost:8000/stream"):
    """
    Stream audio file to WebSocket endpoint in real-time chunks.

    Args:
        file_path: Path to WAV file (16kHz mono recommended)
        ws_url: WebSocket URL
    """
    # Load audio file
    audio, sample_rate = sf.read(file_path, dtype='float32')

    # Ensure mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed (simple linear interpolation)
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate}Hz to 16000Hz")
        duration = len(audio) / sample_rate
        target_length = int(duration * 16000)
        audio = np.interp(
            np.linspace(0, len(audio), target_length),
            np.arange(len(audio)),
            audio
        )
        sample_rate = 16000

    print(f"Loaded audio: {len(audio)/sample_rate:.2f}s @ {sample_rate}Hz")
    print(f"Connecting to {ws_url}...")

    # Chunk parameters
    CHUNK_SIZE_MS = 80  # 80ms chunks for low latency
    CHUNK_SAMPLES = int(sample_rate * CHUNK_SIZE_MS / 1000)  # 1280 samples at 16kHz

    latencies = []
    transcript_count = 0

    try:
        async with websockets.connect(ws_url) as websocket:
            print(f"Connected! Streaming audio in {CHUNK_SIZE_MS}ms chunks...\n")

            # Create tasks for sending and receiving
            async def send_audio():
                """Send audio chunks in real-time."""
                for i in range(0, len(audio), CHUNK_SAMPLES):
                    chunk = audio[i:i + CHUNK_SAMPLES]

                    # Pad last chunk if needed
                    if len(chunk) < CHUNK_SAMPLES:
                        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

                    # Convert to int16 PCM
                    chunk_int16 = (chunk * 32768).astype(np.int16)

                    # Send chunk
                    await websocket.send(chunk_int16.tobytes())

                    # Wait to simulate real-time (80ms)
                    await asyncio.sleep(CHUNK_SIZE_MS / 1000)

                print("\n[Audio streaming complete]")

            async def receive_transcripts():
                """Receive and display transcripts."""
                nonlocal transcript_count
                start_time = time.time()

                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        elapsed = time.time() - start_time

                        import json
                        data = json.loads(response)

                        text = data.get("text", "")
                        is_final = data.get("is_final", False)

                        if text:
                            transcript_count += 1
                            latency = elapsed * 1000  # ms
                            latencies.append(latency)

                            # Display transcript
                            status = "FINAL" if is_final else "INTERIM"
                            print(f"[{elapsed:6.2f}s] [{status:7s}] {text}")

                    except asyncio.TimeoutError:
                        break
                    except websockets.exceptions.ConnectionClosed:
                        break

            # Run both tasks concurrently
            await asyncio.gather(
                send_audio(),
                receive_transcripts()
            )

    except websockets.exceptions.WebSocketException as e:
        print(f"\nWebSocket error: {e}")
        print("\nIs the server running? Start with:")
        print("  uvicorn parakeet_service.main:app --reload")
        return

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Transcripts received: {transcript_count}")

    if latencies:
        print(f"First response:       {latencies[0]:.0f}ms")
        print(f"Average latency:      {np.mean(latencies):.0f}ms")
        print(f"Median latency:       {np.median(latencies):.0f}ms")
        print(f"Min latency:          {np.min(latencies):.0f}ms")
        print(f"Max latency:          {np.max(latencies):.0f}ms")

        # Check target
        if latencies[0] < 300:
            print(f"\n✓ Target achieved: <300ms first response")
        else:
            print(f"\n✗ Target missed: first response {latencies[0]:.0f}ms (target: <300ms)")
    else:
        print("No transcripts received - check server logs")


def generate_test_audio(output_path: str = "test_audio.wav"):
    """Generate a simple test audio file if none provided."""
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile required to generate test audio")
        return None

    # Generate 5 seconds of silence (for testing)
    print(f"Generating test audio: {output_path}")
    sample_rate = 16000
    duration = 5.0
    audio = np.random.randn(int(sample_rate * duration)) * 0.01  # Low noise

    sf.write(output_path, audio, sample_rate)
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_streaming_client.py <audio_file.wav>")
        print("\nOr generate test audio:")
        audio_file = generate_test_audio()
        if not audio_file:
            sys.exit(1)
    else:
        audio_file = sys.argv[1]

    if not Path(audio_file).exists():
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)

    # Run test
    asyncio.run(stream_audio_file(audio_file))


if __name__ == "__main__":
    main()
