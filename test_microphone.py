#!/usr/bin/env python3
"""
Real-time microphone transcription test for Parakeet STT.
Tests local or remote (dstack) endpoints with live audio input.

Usage:
    python test_microphone.py                  # Use local endpoint
    python test_microphone.py --remote         # Use remote endpoint from config
    python test_microphone.py --url wss://...  # Use custom endpoint

Requirements:
    pip install websockets sounddevice numpy
"""
import asyncio
import argparse
import sys
import signal
from pathlib import Path

try:
    import websockets
    import sounddevice as sd
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("\nInstall with: pip install websockets sounddevice numpy")
    sys.exit(1)


class MicrophoneStreamer:
    """Stream microphone audio to WebSocket STT endpoint."""

    def __init__(self, ws_url: str, language: str = "en", sample_rate: int = 16000):
        self.ws_url = ws_url
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_duration = 0.08  # 80ms chunks
        self.chunk_samples = int(sample_rate * self.chunk_duration)
        self.running = False
        self.websocket = None

    async def start(self):
        """Start streaming from microphone."""
        print(f"\n{'='*60}")
        print(f"Parakeet STT - Microphone Test")
        print(f"{'='*60}")
        print(f"Endpoint: {self.ws_url}")
        print(f"Language: {self.language}")
        print(f"Sample rate: {self.sample_rate}Hz")
        print(f"Chunk size: {self.chunk_duration*1000:.0f}ms ({self.chunk_samples} samples)")
        print(f"{'='*60}\n")

        # Connect to WebSocket
        print("Connecting to endpoint...")
        try:
            self.websocket = await websockets.connect(self.ws_url, timeout=10)
            print("✓ Connected!\n")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            print("\nTroubleshooting:")
            print("  - Is the server running?")
            print("  - Is the URL correct?")
            print("  - Check network connectivity")
            return

        # Create audio queue
        audio_queue = asyncio.Queue()

        # Audio callback
        def audio_callback(indata, frames, time, status):
            """Called by sounddevice for each audio chunk."""
            if status:
                print(f"Audio status: {status}")

            # Convert to float32 and queue
            audio = indata[:, 0].copy()  # Mono
            asyncio.create_task(audio_queue.put(audio))

        # Start audio stream
        print("Starting microphone...")
        print("Speak into your microphone. Press Ctrl+C to stop.\n")
        print(f"{'─'*60}\n")

        self.running = True

        try:
            # Open audio stream
            stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_samples,
                callback=audio_callback,
                dtype=np.float32
            )

            with stream:
                # Create tasks
                send_task = asyncio.create_task(self._send_audio(audio_queue))
                recv_task = asyncio.create_task(self._receive_transcripts())

                # Wait for both tasks
                await asyncio.gather(send_task, recv_task)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        except Exception as e:
            print(f"\n\nError: {e}")
        finally:
            self.running = False
            if self.websocket:
                await self.websocket.close()
            print("\nDisconnected.")

    async def _send_audio(self, audio_queue):
        """Send audio chunks to WebSocket."""
        try:
            while self.running:
                # Get audio chunk from queue
                audio_chunk = await audio_queue.get()

                # Convert float32 to int16 PCM
                audio_int16 = (audio_chunk * 32768).astype(np.int16)

                # Send to WebSocket
                await self.websocket.send(audio_int16.tobytes())

        except Exception as e:
            print(f"\nSend error: {e}")
            self.running = False

    async def _receive_transcripts(self):
        """Receive and display transcripts from WebSocket."""
        try:
            while self.running:
                # Receive response
                response = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=1.0
                )

                # Parse JSON
                import json
                data = json.loads(response)

                text = data.get("text", "")
                is_final = data.get("is_final", False)

                if text:
                    # Display transcript
                    status = "[FINAL]" if is_final else "[INTERIM]"
                    print(f"{status:10s} {text}")

        except asyncio.TimeoutError:
            # No response, continue
            if self.running:
                await self._receive_transcripts()
        except Exception as e:
            if self.running:
                print(f"\nReceive error: {e}")
            self.running = False


def load_config():
    """Load test configuration."""
    config_path = Path(__file__).parent / "tests" / "test_config.yml"

    if not config_path.exists():
        return {}

    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        print("Warning: pyyaml not installed, using default config")
        return {}
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        return {}


def get_endpoint_url(args):
    """Get endpoint URL from arguments or config."""
    # Custom URL specified
    if args.url:
        return args.url

    # Load config
    config = load_config()

    # Remote endpoint
    if args.remote:
        remote_cfg = config.get("remote", {})
        url = remote_cfg.get("url")

        if not url:
            print("Error: Remote endpoint not configured in tests/test_config.yml")
            print("\nEdit tests/test_config.yml and set:")
            print("  remote:")
            print("    url: wss://your-deployment.dstack.cloud")
            sys.exit(1)

        return url

    # Local endpoint (default)
    local_cfg = config.get("local", {})
    protocol = local_cfg.get("protocol", "ws")
    host = local_cfg.get("host", "localhost")
    port = local_cfg.get("port", 8000)

    return f"{protocol}://{host}:{port}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time microphone transcription test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test local server
  python test_microphone.py

  # Test remote (dstack) server
  python test_microphone.py --remote

  # Test custom endpoint
  python test_microphone.py --url wss://your-server.com

  # Specify language
  python test_microphone.py --language es --remote
        """
    )

    parser.add_argument(
        "--url",
        type=str,
        help="WebSocket URL (e.g., ws://localhost:8000 or wss://...)"
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use remote endpoint from test_config.yml"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., en, es, fr, de). Leave empty for auto-detect."
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate (default: 16000)"
    )

    args = parser.parse_args()

    # Get endpoint URL
    base_url = get_endpoint_url(args)
    ws_url = f"{base_url}/stream"

    # Prompt for language if not specified
    if args.language is None:
        print("Language Detection")
        print("="*60)
        print("Parakeet TDT supports 25+ languages with auto-detection.")
        print("\nCommon language codes:")
        print("  en (English), es (Spanish), fr (French), de (German),")
        print("  it (Italian), pt (Portuguese), ru (Russian), zh (Chinese),")
        print("  ja (Japanese), ko (Korean), ar (Arabic), hi (Hindi)")
        print()
        language = input("Enter language code (or press Enter for auto-detect): ").strip()

        if not language:
            language = "auto"
            print("Using auto-detection\n")
    else:
        language = args.language

    # Check microphone
    print("\nChecking audio devices...")
    try:
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        print(f"✓ Using microphone: {default_input['name']}")
    except Exception as e:
        print(f"✗ Microphone error: {e}")
        print("\nTroubleshooting:")
        print("  - Check microphone is connected")
        print("  - Check permissions")
        print("  - Try: python -m sounddevice")
        sys.exit(1)

    # Create and start streamer
    streamer = MicrophoneStreamer(
        ws_url=ws_url,
        language=language,
        sample_rate=args.sample_rate
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nInterrupted by user")
        streamer.running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Run
    try:
        asyncio.run(streamer.start())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
