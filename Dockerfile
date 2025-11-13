# Parakeet TDT Streaming STT - CPU-only Dockerfile
# For local development and testing without GPU

FROM python:3.10-slim

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install UV (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY parakeet_service ./parakeet_service
COPY livekit_plugin.py .
COPY test_streaming_client.py .
COPY STREAMING.md .

# Copy environment config
COPY .env.example .env

# Create virtual environment and install dependencies with UV
RUN uv venv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

# Install PyTorch CPU version using UV
RUN uv pip install \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies with UV
RUN uv pip install -e ".[livekit]"

# Set environment variables for CPU mode
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_PRECISION=fp32 \
    DEVICE=cpu \
    LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/healthz')" || exit 1

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "parakeet_service.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
