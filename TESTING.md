# Testing Guide

Comprehensive testing guide for Parakeet STT streaming server.

## Quick Start

### Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
pytest
```

## Test Categories

### 1. Sanity Tests (No Server Required)

Fast unit tests that don't require a running server or model:

```bash
pytest -m sanity
```

**What it tests:**
- Core components (StreamingSession, audio processing)
- Data format conversions
- Configuration loading
- Import checks

**Run time:** ~5 seconds

### 2. Integration Tests (Local Server Required)

Tests that require a running local server:

```bash
# Terminal 1: Start server
uvicorn parakeet_service.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Run tests
pytest -m integration
```

**What it tests:**
- WebSocket connection
- Streaming protocol
- Latency measurements
- Concurrent connections
- Error handling

**Run time:** ~30 seconds

### 3. Remote Tests (dstack Deployment Required)

Tests for deployed instances:

```bash
# Configure remote endpoint in tests/test_config.yml
pytest -m remote
```

**What it tests:**
- Remote connectivity
- End-to-end latency
- Load handling
- Long-running sessions
- Production deployment checks

**Run time:** ~2 minutes

## Configuration

Edit `tests/test_config.yml` to configure endpoints:

```yaml
# Local endpoint
local:
  host: localhost
  port: 8000
  protocol: ws

# Remote endpoint (dstack)
remote:
  url: wss://your-deployment.dstack.cloud
```

## Test Selection

### Run Specific Test Categories

```bash
# Only sanity tests
pytest -m sanity

# Integration + remote
pytest -m "integration or remote"

# Everything except slow tests
pytest -m "not slow"
```

### Run Specific Test Files

```bash
# Sanity tests only
pytest tests/test_sanity.py

# Integration tests only
pytest tests/test_integration.py

# Remote tests only
pytest tests/test_remote.py
```

### Run Specific Test Classes

```bash
# WebSocket connection tests
pytest tests/test_integration.py::TestWebSocketConnection

# Remote streaming tests
pytest tests/test_remote.py::TestRemoteStreaming
```

### Run Specific Tests

```bash
# Single test
pytest tests/test_sanity.py::test_imports

# Test with verbose output
pytest -v tests/test_integration.py::TestLatency::test_first_response_latency
```

## Interactive Testing

### Test with Audio File

```bash
python test_streaming_client.py your_audio.wav
```

**Output:**
```
Loaded audio: 5.23s @ 16000Hz
Connecting to ws://localhost:8000/stream...
Connected! Streaming audio in 80ms chunks...

[  0.24s] [INTERIM] hello
[  0.48s] [INTERIM] hello world
[  1.12s] [FINAL  ] hello world how are you

[Audio streaming complete]

STATISTICS
Transcripts received: 12
First response:       240ms
Average latency:      280ms
Median latency:       270ms
Min latency:          240ms
Max latency:          350ms

✓ Target achieved: <300ms first response
```

### Test with Microphone

```bash
# Local endpoint
python test_microphone.py

# Remote endpoint (dstack)
python test_microphone.py --remote

# Custom endpoint
python test_microphone.py --url wss://your-server.com

# Specify language
python test_microphone.py --language es --remote
```

**Interactive prompts:**
```
Language Detection
============================================================
Parakeet TDT supports 25+ languages with auto-detection.

Common language codes:
  en (English), es (Spanish), fr (French), de (German),
  it (Italian), pt (Portuguese), ru (Russian), zh (Chinese),
  ja (Japanese), ko (Korean), ar (Arabic), hi (Hindi)

Enter language code (or press Enter for auto-detect): en

Checking audio devices...
✓ Using microphone: MacBook Pro Microphone

============================================================
Parakeet STT - Microphone Test
============================================================
Endpoint: ws://localhost:8000/stream
Language: en
Sample rate: 16000Hz
Chunk size: 80ms (1280 samples)
============================================================

Connecting to endpoint...
✓ Connected!

Starting microphone...
Speak into your microphone. Press Ctrl+C to stop.

────────────────────────────────────────────────────────────

[INTERIM]   hello
[INTERIM]   hello this is
[INTERIM]   hello this is a test
[FINAL]     hello this is a test
```

## Test Coverage

### Generate Coverage Report

```bash
pytest --cov=parakeet_service --cov-report=html --cov-report=term
```

View HTML report:
```bash
open htmlcov/index.html
```

### Coverage Targets

- **parakeet_service/streaming_server.py:** ≥80%
- **parakeet_service/main.py:** ≥70%
- **livekit_plugin.py:** ≥75%
- **Overall:** ≥70%

## Parallel Execution

Speed up tests with parallel execution:

```bash
# Auto-detect CPU count
pytest -n auto

# Specific number of workers
pytest -n 4
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt

      - name: Run sanity tests
        run: pytest -m sanity

      - name: Start server
        run: |
          uvicorn parakeet_service.main:app &
          sleep 10

      - name: Run integration tests
        run: pytest -m integration

      - name: Generate coverage
        run: pytest --cov=parakeet_service --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Troubleshooting

### Tests Can't Connect to Server

```bash
# Check server is running
curl http://localhost:8000/healthz

# Check WebSocket endpoint
wscat -c ws://localhost:8000/stream

# Check logs
uvicorn parakeet_service.main:app --log-level debug
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt

# Check PYTHONPATH
export PYTHONPATH=/path/to/parakeet-tdt-0.6b-v2-fastapi:$PYTHONPATH
```

### Microphone Test Errors

```bash
# List audio devices
python -m sounddevice

# Test microphone
python -c "import sounddevice as sd; print(sd.query_devices())"

# Check permissions (macOS)
# System Preferences > Security & Privacy > Microphone
```

### Remote Tests Fail

```bash
# Test connectivity
curl https://your-deployment.dstack.cloud/healthz

# Test WebSocket
wscat -c wss://your-deployment.dstack.cloud/stream

# Check test_config.yml
cat tests/test_config.yml
```

## Test Development

### Adding New Tests

1. Choose appropriate file:
   - `test_sanity.py` - Unit tests, no server
   - `test_integration.py` - Local server tests
   - `test_remote.py` - Remote deployment tests

2. Use appropriate markers:
```python
@pytest.mark.sanity
def test_my_unit_test():
    pass

@pytest.mark.integration
@pytest.mark.asyncio
async def test_my_integration():
    pass

@pytest.mark.remote
@pytest.mark.slow
async def test_my_slow_remote():
    pass
```

3. Use fixtures from `conftest.py`:
```python
def test_with_fixtures(sample_audio_chunk, mock_model):
    # Test code here
    pass
```

### Best Practices

1. **Keep tests fast** - Use mocks for sanity tests
2. **Make tests independent** - Don't rely on test order
3. **Use fixtures** - Share setup code
4. **Clean up resources** - Close connections, delete temp files
5. **Test edge cases** - Empty audio, malformed data, etc.
6. **Document expectations** - Clear assertions with messages

## Performance Benchmarking

### Measure Latency

```bash
python test_streaming_client.py test.wav | grep "latency"
```

### Stress Test

```bash
# Run multiple clients concurrently
for i in {1..10}; do
  python test_streaming_client.py test.wav &
done
wait
```

### Continuous Monitoring

```bash
# Run tests every 5 minutes
watch -n 300 'pytest -m integration --tb=no -q'
```

## Test Metrics

### Key Performance Indicators

| Metric | Target | Command |
|--------|--------|---------|
| Test execution time | <2 min | `pytest --durations=10` |
| Code coverage | ≥70% | `pytest --cov` |
| Integration tests pass rate | 100% | `pytest -m integration` |
| First response latency | <300ms | `python test_streaming_client.py` |
| Remote latency | <2s | `pytest -m remote -v` |

### Monitoring Dashboard

Create a simple dashboard:

```bash
#!/bin/bash
# test_dashboard.sh

echo "=== Parakeet STT Test Dashboard ==="
echo

echo "Sanity Tests:"
pytest -m sanity -q --tb=no
echo

echo "Integration Tests:"
pytest -m integration -q --tb=no
echo

echo "Coverage:"
pytest --cov=parakeet_service --cov-report=term-missing | tail -n 20
echo

echo "Latency:"
python test_streaming_client.py test.wav 2>/dev/null | grep "latency"
```

## FAQ

**Q: Do I need a GPU to run tests?**
A: Sanity tests don't need GPU. Integration/remote tests work with or without GPU (slower without).

**Q: Can I test without NeMo installed?**
A: Sanity tests use mocks and will pass. Integration tests need NeMo.

**Q: How do I test a specific dstack deployment?**
A: Edit `tests/test_config.yml` and set `remote.url`, then run `pytest -m remote`.

**Q: What if microphone test has no output?**
A: The model needs real speech. Test audio (noise) may not produce transcripts.

**Q: How do I debug failing tests?**
A: Run with `-vv --tb=long --log-cli-level=DEBUG` for detailed output.

**Q: Can I run tests without starting the server manually?**
A: Yes, but you'll need to modify tests to use TestClient or start server programmatically.

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [WebSockets Library](https://websockets.readthedocs.io/)
- [SoundDevice](https://python-sounddevice.readthedocs.io/)
