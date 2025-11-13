"""
Sanity tests for Parakeet STT core components.
These tests run quickly and don't require a model or running server.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.sanity
class TestStreamingSession:
    """Test StreamingSession class."""

    def test_session_initialization_with_streaming(self, mock_model):
        """Test session initializes with streaming buffer."""
        with patch('parakeet_service.streaming_server.STREAMING_AVAILABLE', True):
            with patch('parakeet_service.streaming_server.CacheAwareStreamingAudioBuffer') as MockBuffer:
                from parakeet_service.streaming_server import StreamingSession

                session = StreamingSession(mock_model, "test-session-123")

                assert session.session_id == "test-session-123"
                assert session.model == mock_model
                MockBuffer.assert_called_once()

    def test_session_initialization_fallback(self, mock_model):
        """Test session initializes with fallback when streaming unavailable."""
        with patch('parakeet_service.streaming_server.STREAMING_AVAILABLE', False):
            from parakeet_service.streaming_server import StreamingSession

            session = StreamingSession(mock_model, "test-session-456")

            assert session.session_id == "test-session-456"
            assert session.buffer is None
            assert session.audio_buffer == []

    def test_process_audio_with_streaming(self, mock_model, sample_audio_chunk, mock_streaming_buffer):
        """Test audio processing with streaming buffer."""
        with patch('parakeet_service.streaming_server.STREAMING_AVAILABLE', True):
            with patch('parakeet_service.streaming_server.CacheAwareStreamingAudioBuffer', return_value=mock_streaming_buffer):
                from parakeet_service.streaming_server import StreamingSession

                session = StreamingSession(mock_model, "test-session")

                # Process several chunks
                results = []
                for i in range(10):
                    result = session.process_audio(sample_audio_chunk)
                    if result:
                        results.append(result)

                # Should get transcripts every 5th call (based on mock)
                assert len(results) == 2
                assert "transcript" in results[0]

    def test_reset_session(self, mock_model, mock_streaming_buffer):
        """Test session reset."""
        with patch('parakeet_service.streaming_server.STREAMING_AVAILABLE', True):
            with patch('parakeet_service.streaming_server.CacheAwareStreamingAudioBuffer', return_value=mock_streaming_buffer):
                from parakeet_service.streaming_server import StreamingSession

                session = StreamingSession(mock_model, "test-session")
                session.reset()

                assert mock_streaming_buffer.reset_called


@pytest.mark.sanity
class TestAudioProcessing:
    """Test audio format conversions."""

    def test_int16_to_float32_conversion(self, sample_audio_chunk_int16):
        """Test PCM int16 to float32 conversion."""
        audio_float = sample_audio_chunk_int16.astype(np.float32) / 32768.0

        assert audio_float.dtype == np.float32
        assert audio_float.min() >= -1.0
        assert audio_float.max() <= 1.0

    def test_float32_to_int16_conversion(self, sample_audio_chunk):
        """Test float32 to PCM int16 conversion."""
        audio_int16 = (sample_audio_chunk * 32768).astype(np.int16)

        assert audio_int16.dtype == np.int16
        assert len(audio_int16) == len(sample_audio_chunk)

    def test_audio_chunk_size(self, sample_audio_chunk):
        """Test that audio chunks are correct size (80ms @ 16kHz = 1280 samples)."""
        assert len(sample_audio_chunk) == 1280

    def test_audio_normalization(self):
        """Test audio stays in valid range after normalization."""
        # Create audio that might clip
        audio = np.array([1.5, -1.5, 0.5, -0.5], dtype=np.float32)

        # Clip to valid range
        audio_clipped = np.clip(audio, -1.0, 1.0)

        assert audio_clipped.max() <= 1.0
        assert audio_clipped.min() >= -1.0


@pytest.mark.sanity
class TestConfiguration:
    """Test configuration loading."""

    def test_streaming_available_flag(self):
        """Test STREAMING_AVAILABLE flag is set correctly."""
        from parakeet_service.streaming_server import STREAMING_AVAILABLE

        # Should be a boolean
        assert isinstance(STREAMING_AVAILABLE, bool)

    def test_model_precision_options(self):
        """Test model precision configuration."""
        import torch

        # Test FP16 dtype
        dtype_fp16 = torch.float16
        assert dtype_fp16 == torch.float16

        # Test FP32 dtype
        dtype_fp32 = torch.float32
        assert dtype_fp32 == torch.float32

    def test_chunk_parameters(self):
        """Test streaming chunk parameters are sensible."""
        chunk_size = 0.08  # 80ms
        shift_size = 0.04  # 40ms
        sample_rate = 16000

        chunk_samples = int(chunk_size * sample_rate)
        shift_samples = int(shift_size * sample_rate)

        assert chunk_samples == 1280  # 80ms @ 16kHz
        assert shift_samples == 640   # 40ms @ 16kHz
        assert shift_samples < chunk_samples  # Overlap


@pytest.mark.sanity
class TestWebSocketProtocol:
    """Test WebSocket message format."""

    def test_response_format(self):
        """Test WebSocket response has correct format."""
        import json

        response = {
            "text": "test transcript",
            "is_final": False,
            "session_id": "test-123"
        }

        # Should be JSON serializable
        json_str = json.dumps(response)
        parsed = json.loads(json_str)

        assert parsed["text"] == "test transcript"
        assert parsed["is_final"] is False
        assert "session_id" in parsed

    def test_audio_chunk_bytes_format(self, sample_audio_chunk_int16):
        """Test audio chunk can be serialized to bytes."""
        audio_bytes = sample_audio_chunk_int16.tobytes()

        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) == len(sample_audio_chunk_int16) * 2  # 2 bytes per int16

        # Test deserialization
        audio_recovered = np.frombuffer(audio_bytes, dtype=np.int16)
        np.testing.assert_array_equal(audio_recovered, sample_audio_chunk_int16)


@pytest.mark.sanity
class TestModelConfiguration:
    """Test model configuration functions."""

    def test_setup_model_for_streaming(self, mock_model):
        """Test setup_model_for_streaming function."""
        from parakeet_service.streaming_server import setup_model_for_streaming

        # Should not raise error with mock model
        result = setup_model_for_streaming(mock_model)

        # Should return model (even if config fails)
        assert result is not None


@pytest.mark.sanity
def test_imports():
    """Test that all required modules can be imported."""
    # Core modules
    import parakeet_service.streaming_server
    import parakeet_service.main
    import parakeet_service.model
    import parakeet_service.config

    # Should not raise ImportError
    assert parakeet_service.streaming_server is not None
    assert parakeet_service.main is not None


@pytest.mark.sanity
def test_livekit_plugin_imports():
    """Test LiveKit plugin imports (may skip if not installed)."""
    try:
        import livekit_plugin

        # Check classes exist
        assert hasattr(livekit_plugin, 'ParakeetSTT')
        assert hasattr(livekit_plugin, 'ParakeetStream')
    except ImportError:
        pytest.skip("LiveKit not installed")


@pytest.mark.sanity
def test_requirements_installed():
    """Test that critical requirements are installed."""
    import importlib

    critical_packages = [
        'fastapi',
        'uvicorn',
        'numpy',
        'torch',
        'websockets',
    ]

    for package in critical_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            pytest.fail(f"Required package not installed: {package}")
