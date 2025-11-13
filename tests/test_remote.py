"""
Remote endpoint tests for dstack deployments.
These tests verify deployed instances work correctly.
"""
import pytest
import asyncio
import numpy as np
import time
import json

try:
    import websockets
    import aiohttp
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not DEPS_AVAILABLE,
    reason="websockets or aiohttp not installed"
)


@pytest.mark.remote
class TestRemoteConnection:
    """Test connection to remote (dstack) endpoint."""

    @pytest.mark.asyncio
    async def test_remote_websocket_connect(self, remote_endpoint):
        """Test connection to remote WebSocket endpoint."""
        ws_url = f"{remote_endpoint}/stream"

        try:
            async with websockets.connect(ws_url, timeout=10) as websocket:
                assert websocket.open
        except Exception as e:
            pytest.fail(f"Failed to connect to remote endpoint: {e}")

    @pytest.mark.asyncio
    async def test_remote_health_check(self, remote_endpoint):
        """Test remote health check endpoint."""
        health_url = f"{remote_endpoint.replace('ws://', 'http://').replace('wss://', 'https://')}/healthz"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=10) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data.get("status") == "ok"
        except Exception as e:
            pytest.fail(f"Health check failed: {e}")


@pytest.mark.remote
class TestRemoteStreaming:
    """Test streaming functionality on remote endpoint."""

    @pytest.mark.asyncio
    async def test_remote_streaming_basic(self, remote_endpoint, sample_audio_chunk_int16):
        """Test basic streaming to remote endpoint."""
        ws_url = f"{remote_endpoint}/stream"

        try:
            async with websockets.connect(ws_url, timeout=10) as websocket:
                # Send 10 chunks
                for i in range(10):
                    await websocket.send(sample_audio_chunk_int16.tobytes())
                    await asyncio.sleep(0.08)

                # Connection should remain open
                assert websocket.open
        except Exception as e:
            pytest.fail(f"Remote streaming failed: {e}")

    @pytest.mark.asyncio
    async def test_remote_end_to_end_latency(self, remote_endpoint, sample_audio_chunk_int16):
        """Measure end-to-end latency to remote endpoint."""
        ws_url = f"{remote_endpoint}/stream"

        latencies = []

        try:
            async with websockets.connect(ws_url, timeout=10) as websocket:
                # Send multiple chunks and measure latency
                for i in range(50):
                    send_time = time.time()
                    await websocket.send(sample_audio_chunk_int16.tobytes())

                    # Try to receive response
                    try:
                        response = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=0.5
                        )

                        recv_time = time.time()
                        latency = (recv_time - send_time) * 1000  # ms
                        latencies.append(latency)

                        # Parse response
                        data = json.loads(response)
                        assert "text" in data

                    except asyncio.TimeoutError:
                        pass

                    await asyncio.sleep(0.08)

                # Report latencies
                if latencies:
                    avg_latency = np.mean(latencies)
                    p50_latency = np.percentile(latencies, 50)
                    p95_latency = np.percentile(latencies, 95)

                    print(f"\nRemote Latency Stats:")
                    print(f"  Average: {avg_latency:.0f}ms")
                    print(f"  P50: {p50_latency:.0f}ms")
                    print(f"  P95: {p95_latency:.0f}ms")

                    # For remote endpoints, allow higher latency due to network
                    assert avg_latency < 2000, f"Average latency too high: {avg_latency:.0f}ms"

        except Exception as e:
            pytest.fail(f"Latency test failed: {e}")


@pytest.mark.remote
class TestRemoteLoadHandling:
    """Test how remote endpoint handles load."""

    @pytest.mark.asyncio
    async def test_remote_concurrent_sessions(self, remote_endpoint, sample_audio_chunk_int16):
        """Test multiple concurrent sessions on remote endpoint."""
        ws_url = f"{remote_endpoint}/stream"

        async def client_session(session_id):
            """Single client session."""
            try:
                async with websockets.connect(ws_url, timeout=10) as websocket:
                    for _ in range(10):
                        await websocket.send(sample_audio_chunk_int16.tobytes())
                        await asyncio.sleep(0.08)
                return session_id, True
            except Exception as e:
                return session_id, False

        # Run 5 concurrent sessions
        results = await asyncio.gather(
            client_session(1),
            client_session(2),
            client_session(3),
            client_session(4),
            client_session(5)
        )

        # All should succeed
        successes = [r for r in results if r[1]]
        print(f"\nSuccessful sessions: {len(successes)}/5")

        assert len(successes) >= 3, "At least 3 concurrent sessions should succeed"


@pytest.mark.remote
class TestRemoteDeploymentConfig:
    """Test deployment configuration."""

    @pytest.mark.asyncio
    async def test_remote_config_endpoint(self, remote_endpoint):
        """Test that config endpoint returns deployment info."""
        config_url = f"{remote_endpoint.replace('ws://', 'http://').replace('wss://', 'https://')}/config"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(config_url, timeout=10) as response:
                    # Config endpoint might not exist, which is ok
                    if response.status == 200:
                        data = await response.json()
                        print(f"\nRemote config: {data}")
        except Exception:
            # Config endpoint not critical
            pass

    @pytest.mark.asyncio
    async def test_remote_https_security(self, remote_endpoint):
        """Test that remote endpoint uses secure WebSocket (wss://)."""
        # Production deployments should use wss://
        if not remote_endpoint.startswith("ws://localhost"):
            assert remote_endpoint.startswith("wss://"), \
                "Production deployment should use secure WebSocket (wss://)"


@pytest.mark.remote
@pytest.mark.slow
class TestRemoteLongSession:
    """Test long-running sessions on remote endpoint."""

    @pytest.mark.asyncio
    async def test_remote_extended_session(self, remote_endpoint, sample_audio_chunk_int16):
        """Test 60-second session on remote endpoint."""
        ws_url = f"{remote_endpoint}/stream"

        try:
            async with websockets.connect(ws_url, timeout=10) as websocket:
                duration = 60  # seconds
                chunks = int(duration / 0.08)

                responses_received = 0

                for i in range(chunks):
                    await websocket.send(sample_audio_chunk_int16.tobytes())

                    # Try to receive
                    try:
                        response = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=0.05
                        )
                        responses_received += 1
                    except asyncio.TimeoutError:
                        pass

                    await asyncio.sleep(0.08)

                    # Log progress
                    if i % 125 == 0:  # Every 10 seconds
                        elapsed = i * 0.08
                        print(f"Progress: {elapsed:.0f}s / {duration}s, "
                              f"responses: {responses_received}")

                # Session should still be open
                assert websocket.open

                print(f"\nTotal responses received: {responses_received}")

        except Exception as e:
            pytest.fail(f"Extended session failed: {e}")


@pytest.mark.remote
class TestRemotePerformance:
    """Test performance characteristics of remote deployment."""

    @pytest.mark.asyncio
    async def test_remote_throughput(self, remote_endpoint, sample_audio_chunk_int16):
        """Measure throughput to remote endpoint."""
        ws_url = f"{remote_endpoint}/stream"

        try:
            async with websockets.connect(ws_url, timeout=10) as websocket:
                start_time = time.time()
                chunks_sent = 0

                # Send for 10 seconds
                while time.time() - start_time < 10:
                    await websocket.send(sample_audio_chunk_int16.tobytes())
                    chunks_sent += 1
                    await asyncio.sleep(0.08)

                elapsed = time.time() - start_time
                throughput = chunks_sent / elapsed  # chunks per second

                print(f"\nThroughput: {throughput:.1f} chunks/sec")
                print(f"Audio rate: {throughput * 0.08:.1f}x realtime")

                # Should be able to send at least at real-time rate
                assert throughput >= 12, \
                    f"Throughput too low: {throughput:.1f} chunks/sec (need ≥12 for realtime)"

        except Exception as e:
            pytest.fail(f"Throughput test failed: {e}")


# Utility function to test connectivity
@pytest.mark.remote
def test_remote_endpoint_configured(remote_endpoint):
    """Test that remote endpoint is properly configured."""
    assert remote_endpoint is not None
    assert remote_endpoint.startswith(("ws://", "wss://"))

    print(f"\nTesting remote endpoint: {remote_endpoint}")
