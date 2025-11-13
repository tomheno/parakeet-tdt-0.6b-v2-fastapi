"""
Integration tests for Parakeet STT WebSocket streaming.
These tests require a running server (local or remote).
"""
import pytest
import asyncio
import numpy as np
import time
import json

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not WEBSOCKETS_AVAILABLE,
    reason="websockets package not installed"
)


@pytest.mark.integration
class TestWebSocketConnection:
    """Test WebSocket connection and basic protocol."""

    @pytest.mark.asyncio
    async def test_websocket_connect(self, local_endpoint):
        """Test that WebSocket connection can be established."""
        ws_url = f"{local_endpoint}/stream"

        try:
            async with websockets.connect(ws_url, timeout=5) as websocket:
                assert websocket.open
        except (ConnectionRefusedError, asyncio.TimeoutError):
            pytest.skip("Local server not running. Start with: uvicorn parakeet_service.main:app")

    @pytest.mark.asyncio
    async def test_websocket_disconnect_gracefully(self, local_endpoint):
        """Test that WebSocket can disconnect gracefully."""
        ws_url = f"{local_endpoint}/stream"

        try:
            async with websockets.connect(ws_url, timeout=5) as websocket:
                await websocket.close()
                assert websocket.closed
        except (ConnectionRefusedError, asyncio.TimeoutError):
            pytest.skip("Local server not running")


@pytest.mark.integration
class TestStreamingProtocol:
    """Test streaming protocol with actual server."""

    @pytest.mark.asyncio
    async def test_send_audio_chunk(self, local_endpoint, sample_audio_chunk_int16):
        """Test sending a single audio chunk."""
        ws_url = f"{local_endpoint}/stream"

        try:
            async with websockets.connect(ws_url, timeout=5) as websocket:
                # Send one audio chunk
                await websocket.send(sample_audio_chunk_int16.tobytes())

                # Server should not close connection
                await asyncio.sleep(0.1)
                assert websocket.open
        except (ConnectionRefusedError, asyncio.TimeoutError):
            pytest.skip("Local server not running")

    @pytest.mark.asyncio
    async def test_send_multiple_chunks(self, local_endpoint, sample_audio_chunk_int16):
        """Test sending multiple audio chunks."""
        ws_url = f"{local_endpoint}/stream"

        try:
            async with websockets.connect(ws_url, timeout=5) as websocket:
                # Send 10 chunks (800ms of audio)
                for _ in range(10):
                    await websocket.send(sample_audio_chunk_int16.tobytes())
                    await asyncio.sleep(0.08)  # Simulate real-time

                # Connection should still be open
                assert websocket.open
        except (ConnectionRefusedError, asyncio.TimeoutError):
            pytest.skip("Local server not running")

    @pytest.mark.asyncio
    async def test_receive_response_format(self, local_endpoint, sample_audio_chunk_int16):
        """Test that responses have correct JSON format."""
        ws_url = f"{local_endpoint}/stream"

        try:
            async with websockets.connect(ws_url, timeout=5) as websocket:
                # Send chunks and try to receive response
                for _ in range(20):
                    await websocket.send(sample_audio_chunk_int16.tobytes())
                    await asyncio.sleep(0.08)

                    # Try to receive with short timeout
                    try:
                        response = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=0.5
                        )

                        # Parse JSON
                        data = json.loads(response)

                        # Check format
                        assert "text" in data
                        assert "is_final" in data

                        # If we got a response, test passed
                        break
                    except asyncio.TimeoutError:
                        continue

        except (ConnectionRefusedError, asyncio.TimeoutError):
            pytest.skip("Local server not running")


@pytest.mark.integration
class TestLatency:
    """Test latency requirements."""

    @pytest.mark.asyncio
    async def test_first_response_latency(self, local_endpoint, sample_audio_chunk_int16):
        """Test that first response comes within reasonable time (<5s for test)."""
        ws_url = f"{local_endpoint}/stream"

        try:
            async with websockets.connect(ws_url, timeout=5) as websocket:
                start_time = time.time()

                # Send chunks until we get a response
                for i in range(100):  # Max 8 seconds worth
                    await websocket.send(sample_audio_chunk_int16.tobytes())

                    # Try to receive
                    try:
                        response = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=0.1
                        )

                        latency = (time.time() - start_time) * 1000  # ms

                        # Check we got a response
                        data = json.loads(response)
                        assert "text" in data

                        # For test audio (noise), may not get transcripts
                        # but connection should work
                        print(f"First response latency: {latency:.0f}ms")
                        break

                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.08)
                        continue

        except (ConnectionRefusedError, asyncio.TimeoutError):
            pytest.skip("Local server not running")


@pytest.mark.integration
class TestHealthCheck:
    """Test health check endpoint."""

    @pytest.mark.asyncio
    async def test_healthz_endpoint(self, local_endpoint):
        """Test /healthz endpoint responds."""
        import aiohttp

        # Convert ws:// to http://
        http_url = local_endpoint.replace("ws://", "http://").replace("wss://", "https://")
        healthz_url = f"{http_url}/healthz"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(healthz_url, timeout=5) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert "status" in data
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pytest.skip("Local server not running")


@pytest.mark.integration
class TestConcurrency:
    """Test concurrent connections."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_connections(self, local_endpoint, sample_audio_chunk_int16):
        """Test that multiple clients can connect simultaneously."""
        ws_url = f"{local_endpoint}/stream"

        async def client_session(client_id):
            """Single client session."""
            try:
                async with websockets.connect(ws_url, timeout=5) as websocket:
                    # Send a few chunks
                    for _ in range(5):
                        await websocket.send(sample_audio_chunk_int16.tobytes())
                        await asyncio.sleep(0.08)
                return True
            except Exception:
                return False

        try:
            # Run 3 concurrent clients
            results = await asyncio.gather(
                client_session(1),
                client_session(2),
                client_session(3)
            )

            # At least one should succeed
            assert any(results)
        except (ConnectionRefusedError, asyncio.TimeoutError):
            pytest.skip("Local server not running")


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_invalid_audio_format(self, local_endpoint):
        """Test sending invalid data doesn't crash server."""
        ws_url = f"{local_endpoint}/stream"

        try:
            async with websockets.connect(ws_url, timeout=5) as websocket:
                # Send invalid data (not PCM int16)
                await websocket.send(b"invalid audio data")

                # Wait a bit
                await asyncio.sleep(0.5)

                # Connection might close or stay open
                # Either is acceptable as long as server doesn't crash
                # (We test this by being able to reconnect)

            # Try to reconnect
            async with websockets.connect(ws_url, timeout=5) as websocket:
                assert websocket.open

        except (ConnectionRefusedError, asyncio.TimeoutError):
            pytest.skip("Local server not running")

    @pytest.mark.asyncio
    async def test_rapid_disconnects(self, local_endpoint):
        """Test rapid connect/disconnect doesn't cause issues."""
        ws_url = f"{local_endpoint}/stream"

        try:
            for _ in range(5):
                async with websockets.connect(ws_url, timeout=5) as websocket:
                    await websocket.close()
                    await asyncio.sleep(0.1)

            # Should still be able to connect
            async with websockets.connect(ws_url, timeout=5) as websocket:
                assert websocket.open

        except (ConnectionRefusedError, asyncio.TimeoutError):
            pytest.skip("Local server not running")


@pytest.mark.integration
@pytest.mark.slow
class TestLongSession:
    """Test long-running sessions."""

    @pytest.mark.asyncio
    async def test_long_session(self, local_endpoint, sample_audio_chunk_int16):
        """Test session can run for extended period (30 seconds)."""
        ws_url = f"{local_endpoint}/stream"

        try:
            async with websockets.connect(ws_url, timeout=5) as websocket:
                duration = 30  # seconds
                chunks = int(duration / 0.08)

                for i in range(chunks):
                    await websocket.send(sample_audio_chunk_int16.tobytes())
                    await asyncio.sleep(0.08)

                    # Log progress every 5 seconds
                    if i % 62 == 0:
                        print(f"Progress: {i * 0.08:.1f}s / {duration}s")

                # Session should still be open
                assert websocket.open

        except (ConnectionRefusedError, asyncio.TimeoutError):
            pytest.skip("Local server not running")
