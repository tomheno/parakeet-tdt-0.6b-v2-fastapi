# Build on Mac → Deploy on Remote GPU

Complete guide for building Docker images on Mac (M1/M2/Intel) and deploying to remote GPU with dstack.

## Prerequisites

- Docker Desktop for Mac
- dstack CLI: `uv tool install dstack`
- Registry account (GitHub Container Registry, Docker Hub, etc.)

## Step 1: Setup Registry

### Option A: GitHub Container Registry (Recommended)

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

# Or create personal access token at:
# https://github.com/settings/tokens
# Scopes needed: write:packages, read:packages
```

### Option B: Docker Hub

```bash
# Login to Docker Hub
docker login

# Or with token
echo $DOCKER_TOKEN | docker login -u YOUR_USERNAME --password-stdin
```

## Step 2: Build Multi-Architecture Image on Mac

**Important:** Build for `linux/amd64` since remote GPUs use x86_64 architecture.

```bash
# Enable buildx (multi-platform builds)
docker buildx create --use --name multiarch

# Build and push for linux/amd64 (GPU servers)
docker buildx build \
  --platform linux/amd64 \
  --file Dockerfile.gpu \
  --tag ghcr.io/YOUR_USERNAME/parakeet-streaming-stt:latest \
  --tag ghcr.io/YOUR_USERNAME/parakeet-streaming-stt:v1.0.0 \
  --push \
  .

# Or for Docker Hub:
docker buildx build \
  --platform linux/amd64 \
  --file Dockerfile.gpu \
  --tag YOUR_USERNAME/parakeet-streaming-stt:latest \
  --push \
  .
```

### Why `linux/amd64`?

- Mac M1/M2: `linux/arm64` (won't work on GPU servers)
- Mac Intel: `linux/amd64` (but better to specify explicitly)
- GPU servers: `linux/amd64` (NVIDIA GPUs on x86_64)

### Build Options

**Fast build (local testing):**
```bash
# Just build, don't push
docker buildx build \
  --platform linux/amd64 \
  --file Dockerfile.gpu \
  --tag parakeet-stt:local \
  --load \
  .

# Test locally (if you have NVIDIA GPU)
docker run --gpus all -p 8000:8000 parakeet-stt:local
```

**Production build:**
```bash
# Build with cache optimization
docker buildx build \
  --platform linux/amd64 \
  --file Dockerfile.gpu \
  --tag ghcr.io/YOUR_USERNAME/parakeet-streaming-stt:$(git rev-parse --short HEAD) \
  --tag ghcr.io/YOUR_USERNAME/parakeet-streaming-stt:latest \
  --cache-from type=registry,ref=ghcr.io/YOUR_USERNAME/parakeet-streaming-stt:buildcache \
  --cache-to type=registry,ref=ghcr.io/YOUR_USERNAME/parakeet-streaming-stt:buildcache,mode=max \
  --push \
  .
```

## Step 3: Verify Image

```bash
# Check image in registry
docker manifest inspect ghcr.io/YOUR_USERNAME/parakeet-streaming-stt:latest

# Should show: "architecture": "amd64"
```

## Step 4: Update dstack Config

Edit `.dstack-registry.yml`:

```yaml
type: service
name: parakeet-streaming-stt-gpu

# Your registry image
image: ghcr.io/YOUR_USERNAME/parakeet-streaming-stt:latest

# ... rest of config
```

## Step 5: Deploy with dstack

```bash
# Initialize dstack (first time only)
dstack init

# Deploy using registry config
dstack run -f .dstack-registry.yml

# Check status
dstack ps

# Get logs
dstack logs parakeet-streaming-stt-gpu

# Get endpoint URL
dstack run ls
```

## Step 6: Test Remote Deployment

```bash
# Test health check
curl https://your-dstack-url.com/healthz

# Test with microphone
uv run python test_microphone.py --url wss://your-dstack-url.com/stream

# Test with audio file
uv run python test_streaming_client.py audio.wav
```

## Common Issues

### Issue: "exec format error"

**Problem:** Built for wrong architecture (arm64 on Mac M1/M2)

**Solution:**
```bash
# Always specify --platform linux/amd64
docker buildx build --platform linux/amd64 ...
```

### Issue: Build is slow on Mac

**Problem:** Emulating amd64 on arm64 Mac

**Solutions:**
1. Use build cache (see production build above)
2. Build less frequently, deploy same image multiple times
3. Use GitHub Actions to build (see below)

### Issue: Registry authentication failed

**Solutions:**
```bash
# GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Docker Hub
docker login

# Check logged in
docker info | grep Username
```

## Automated Builds with GitHub Actions

**Recommended:** Let GitHub Actions build on Linux (faster than Mac)

Create `.github/workflows/build.yml`:

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main, develop]
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v3

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Docker metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ghcr.io/${{ github.repository_owner }}/parakeet-streaming-stt
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          file: Dockerfile.gpu
          platforms: linux/amd64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=registry,ref=ghcr.io/${{ github.repository_owner }}/parakeet-streaming-stt:buildcache
          cache-to: type=registry,ref=ghcr.io/${{ github.repository_owner }}/parakeet-streaming-stt:buildcache,mode=max
```

Then just push to GitHub:
```bash
git push origin main
# GitHub Actions builds and pushes automatically
```

## Registry Comparison

| Registry | Free Tier | Private Images | Speed | Best For |
|----------|-----------|----------------|-------|----------|
| GitHub Container Registry | Unlimited public | 500MB private | Fast | Open source |
| Docker Hub | Unlimited public | 1 private | Fast | Public projects |
| AWS ECR | 500MB/month | Unlimited | Fastest | AWS users |
| Google Artifact Registry | 500MB/month | Unlimited | Fastest | GCP users |

## Cost Optimization

### Reduce Build Time

```bash
# Use buildkit cache
export DOCKER_BUILDKIT=1

# Multi-stage builds (already in Dockerfile.gpu)
# Cache dependencies layer separately from code
```

### Reduce Image Size

```bash
# Check image size
docker images | grep parakeet

# Should be ~5-8GB (includes PyTorch + NeMo + model)
```

### Reduce Registry Storage

```bash
# Delete old tags
docker image prune -a

# Use specific version tags instead of :latest for every push
# ghcr.io/user/app:v1.0.0 instead of :latest
```

## Development Workflow

```bash
# 1. Develop locally with UV
uv sync --extra all
uv run uvicorn parakeet_service.main:app --reload

# 2. Test changes
uv run pytest -m sanity

# 3. Build and push when ready
docker buildx build \
  --platform linux/amd64 \
  --file Dockerfile.gpu \
  --tag ghcr.io/YOUR_USERNAME/parakeet-streaming-stt:dev \
  --push \
  .

# 4. Deploy to dstack
dstack run -f .dstack-registry.yml

# 5. Test remote
uv run python test_microphone.py --url wss://your-url.com/stream

# 6. Tag for production
docker buildx build \
  --platform linux/amd64 \
  --file Dockerfile.gpu \
  --tag ghcr.io/YOUR_USERNAME/parakeet-streaming-stt:v1.0.0 \
  --tag ghcr.io/YOUR_USERNAME/parakeet-streaming-stt:latest \
  --push \
  .
```

## Quick Reference

```bash
# Build on Mac for GPU servers
docker buildx build --platform linux/amd64 -f Dockerfile.gpu -t ghcr.io/USER/app:latest --push .

# Deploy with dstack
dstack run -f .dstack-registry.yml

# Check status
dstack ps

# Get logs
dstack logs parakeet-streaming-stt-gpu

# Stop
dstack stop parakeet-streaming-stt-gpu

# Delete
dstack delete parakeet-streaming-stt-gpu
```

## Troubleshooting

```bash
# Check Docker buildx
docker buildx ls

# Create builder if missing
docker buildx create --use --name multiarch

# Check build platform
docker buildx inspect --bootstrap

# Verify image architecture
docker manifest inspect ghcr.io/USER/app:latest | grep architecture

# Test image locally (if you have GPU)
docker run --gpus all -p 8000:8000 ghcr.io/USER/app:latest
```

## Next Steps

1. Build image on Mac
2. Push to GitHub Container Registry
3. Update `.dstack-registry.yml` with your image URL
4. Deploy with `dstack run -f .dstack-registry.yml`
5. Test with `test_microphone.py --url wss://your-url.com/stream`
6. Scale with dstack autoscaling
7. Monitor with dstack metrics

## Support

- dstack docs: https://dstack.ai/docs
- Docker buildx: https://docs.docker.com/buildx/
- GitHub Container Registry: https://docs.github.com/en/packages
