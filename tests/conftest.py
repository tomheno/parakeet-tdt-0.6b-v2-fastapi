"""
Pytest configuration and shared fixtures for Parakeet STT tests.
"""
import pytest
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any

# Load test configuration
TEST_CONFIG_PATH = Path(__file__).parent / "test_config.yml"


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Load test configuration from test_config.yml."""
    if TEST_CONFIG_PATH.exists():
        with open(TEST_CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {
        "local": {
            "host": "localhost",
            "port": 8000,
            "protocol": "ws"
        }
    }


@pytest.fixture(scope="session")
def local_endpoint(test_config):
    """Local endpoint URL for testing."""
    cfg = test_config.get("local", {})
    protocol = cfg.get("protocol", "ws")
    host = cfg.get("host", "localhost")
    port = cfg.get("port", 8000)
    return f"{protocol}://{host}:{port}"


@pytest.fixture(scope="session")
def remote_endpoint(test_config):
    """Remote endpoint URL (dstack) for testing."""
    cfg = test_config.get("remote", {})
    if not cfg:
        pytest.skip("Remote endpoint not configured in test_config.yml")
    return cfg.get("url")


@pytest.fixture
def sample_audio_chunk():
    """Generate a single 80ms audio chunk (16kHz, mono, float32)."""
    sample_rate = 16000
    duration = 0.08  # 80ms
    samples = int(sample_rate * duration)  # 1280 samples

    # Generate low-amplitude noise (to simulate silence/background)
    audio = np.random.randn(samples).astype(np.float32) * 0.01
    return audio


@pytest.fixture
def sample_audio_chunk_int16():
    """Generate a single 80ms audio chunk as int16 PCM."""
    sample_rate = 16000
    duration = 0.08
    samples = int(sample_rate * duration)

    audio_float = np.random.randn(samples).astype(np.float32) * 0.01
    audio_int16 = (audio_float * 32768).astype(np.int16)
    return audio_int16


@pytest.fixture
def sample_audio_file(tmp_path):
    """Generate a test audio file (WAV, 16kHz, mono, 2 seconds)."""
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not installed")

    sample_rate = 16000
    duration = 2.0
    samples = int(sample_rate * duration)

    # Generate low-amplitude noise
    audio = np.random.randn(samples).astype(np.float32) * 0.01

    # Write to temp file
    audio_path = tmp_path / "test_audio.wav"
    sf.write(audio_path, audio, sample_rate)

    return str(audio_path)


@pytest.fixture
def mock_model():
    """Mock NeMo ASR model for unit tests."""
    class MockModel:
        def __init__(self):
            self.device = "cpu"
            self.dtype = np.float32

        def transcribe(self, audio_files, batch_size=1):
            """Mock transcribe method."""
            return ["mock transcript"] * len(audio_files)

        def eval(self):
            return self

        def cuda(self):
            self.device = "cuda"
            return self

        def half(self):
            self.dtype = np.float16
            return self

        def change_attention_model(self, **kwargs):
            pass

        def parameters(self):
            import torch
            return iter([torch.tensor([1.0])])

    return MockModel()


@pytest.fixture
def mock_streaming_buffer():
    """Mock CacheAwareStreamingAudioBuffer for unit tests."""
    class MockStreamingBuffer:
        def __init__(self, **kwargs):
            self.reset_called = False
            self.infer_count = 0

        def infer_signal(self, audio_chunk):
            """Mock inference - returns transcript every 5th call."""
            self.infer_count += 1
            if self.infer_count % 5 == 0:
                return f"transcript {self.infer_count // 5}"
            return ""

        def reset(self):
            self.reset_called = True
            self.infer_count = 0

    return MockStreamingBuffer()


# Markers for different test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "sanity: Sanity tests that run quickly without model"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that require running server"
    )
    config.addinivalue_line(
        "markers", "remote: Tests that require remote (dstack) endpoint"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
