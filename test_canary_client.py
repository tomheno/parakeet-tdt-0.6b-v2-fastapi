#!/usr/bin/env python3
"""
Test client for the Canary 1B V2 streaming STT / translation service.

Usage
-----
    # Transcribe English audio
    python test_canary_client.py audio.wav

    # Transcribe French audio
    python test_canary_client.py audio_fr.wav --source-lang fr --target-lang fr

    # Translate French audio to English
    python test_canary_client.py audio_fr.wav --source-lang fr --target-lang en

    # Point at a remote server
    python test_canary_client.py audio.wav --url ws://my-server:8000/stream
"""

import argparse
import asyncio
import json
import sys
import time

import numpy as np
import soundfile as sf

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")


async def stream_file(
    file_path: str,
    url: str,
    source_lang: str,
    target_lang: str,
    chunk_ms: int = 80,
):
    sample_rate = 16000
    chunk_samples = int(sample_rate * chunk_ms / 1000)

    # Load and resample audio
    audio, sr = sf.read(file_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        try:
            import torchaudio

            audio_t = __import__("torch").from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, sample_rate)
            audio = audio_t.squeeze(0).numpy()
        except ImportError:
            sys.exit(f"Audio is {sr} Hz — install torchaudio for resampling")

    audio_int16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)

    print(f"File     : {file_path}")
    print(f"Duration : {len(audio) / sample_rate:.2f}s")
    print(f"Language : src={source_lang}  tgt={target_lang}")
    mode = "ASR" if source_lang == target_lang else "Translation"
    print(f"Mode     : {mode}")
    print(f"Server   : {url}")
    print("-" * 60)

    transcripts = []
    latencies = []
    t_start = None

    async with websockets.connect(url) as ws:
        # 1. Send language config
        await ws.send(json.dumps({
            "source_lang": source_lang,
            "target_lang": target_lang,
        }))

        # Wait for ready
        ready = json.loads(await ws.recv())
        if ready.get("status") != "ready":
            print(f"Unexpected server response: {ready}")
            return
        print(f"Session  : {ready['session_id']}")
        print("-" * 60)

        t_start = time.time()

        # 2. Stream audio
        async def send_audio():
            for i in range(0, len(audio_int16), chunk_samples):
                chunk = audio_int16[i : i + chunk_samples]
                await ws.send(chunk.tobytes())
                await asyncio.sleep(chunk_ms / 1000)

            # End utterance after all audio sent
            await asyncio.sleep(0.3)
            await ws.send(json.dumps({"action": "end_utterance"}))

        async def recv_transcripts():
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                except asyncio.TimeoutError:
                    break
                msg = json.loads(raw)
                if "text" in msg:
                    now = time.time()
                    latency_ms = (now - t_start) * 1000
                    latencies.append(latency_ms)
                    transcripts.append(msg)
                    tag = "FINAL  " if msg.get("is_final") else "INTERIM"
                    print(f"  [{tag}] ({latency_ms:7.0f}ms) {msg['text']}")
                    if msg.get("is_final"):
                        break
                elif msg.get("status") == "reset":
                    pass

        send_task = asyncio.create_task(send_audio())
        recv_task = asyncio.create_task(recv_transcripts())
        await asyncio.gather(send_task, recv_task)

    # Stats
    print("-" * 60)
    print("Statistics:")
    print(f"  Transcripts received : {len(transcripts)}")
    if latencies:
        print(f"  First response       : {latencies[0]:.0f}ms")
        print(f"  Average latency      : {sum(latencies)/len(latencies):.0f}ms")
        print(f"  Min / Max            : {min(latencies):.0f}ms / {max(latencies):.0f}ms")
    finals = [t for t in transcripts if t.get("is_final")]
    if finals:
        print(f"  Final text           : {finals[-1]['text']}")


def main():
    parser = argparse.ArgumentParser(description="Canary 1B V2 streaming test client")
    parser.add_argument("audio_file", help="Path to audio file (WAV/FLAC)")
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/stream",
        help="WebSocket endpoint URL",
    )
    parser.add_argument("--source-lang", default="en", help="Source language code")
    parser.add_argument("--target-lang", default="en", help="Target language code")
    parser.add_argument("--chunk-ms", type=int, default=80, help="Chunk size in ms")
    args = parser.parse_args()

    asyncio.run(
        stream_file(
            args.audio_file,
            args.url,
            args.source_lang,
            args.target_lang,
            args.chunk_ms,
        )
    )


if __name__ == "__main__":
    main()
