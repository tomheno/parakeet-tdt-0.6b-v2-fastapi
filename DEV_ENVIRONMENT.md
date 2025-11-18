# Dev Environment with dstack

Fast iteration without rebuilding Docker images.

## Quick Start

### 1. Launch Dev Environment
```bash
dstack apply -f dev.dstack.yml
```

Select an offer and dstack will:
- Provision GPU instance with Python 3.11 + NVCC
- Install UV and all dependencies
- Mount your local repository
- Open VS Code with remote connection

### 2. Open in VS Code
Click the `vscode://` link from the output or use:
```bash
# From the dstack output
code --remote ssh-remote+<run-name> /workflow
```

### 3. Run the Server
In the VS Code terminal:
```bash
# Run the streaming server
uv run uvicorn parakeet_service.main:app --host 0.0.0.0 --port 8000

# Or run tests
uv run pytest tests/test_sanity.py -v
uv run pytest tests/test_integration.py -v
```

### 4. Test from Your Local Machine
The dev environment forwards port 8000, so you can test from your local machine:
```bash
# Test with microphone (on your local machine)
python test_microphone.py --url ws://<dstack-instance-ip>:8000/stream

# Or use the test client
python parakeet_service/test_streaming_client.py
```

## Configuration Details

**Resources:**
- GPU: 16GB (enough for Parakeet TDT 0.6B fp16)
- Memory: 16GB+ RAM
- Disk: 50GB
- Spot instances: Auto (saves ~70% cost)

**Features:**
- Auto-retry on interruption/no-capacity (30min)
- Auto-stop after 2 hours inactivity
- All dependencies pre-installed with UV
- Local repo mounted to `/workflow`

## Management Commands

```bash
# List running dev environments
dstack ps

# Monitor status
dstack ps --watch

# Attach to running environment
dstack attach <run-name>

# Stop environment
dstack stop <run-name>

# View logs
dstack logs <run-name>
```

## Cost Optimization

**Spot instances** reduce costs by ~70% but can be interrupted:
- Dev environment auto-retries for 30 minutes
- Your work is saved (mounted from local repo)
- VS Code reconnects automatically

**Inactivity timeout** stops environment after 2 hours of:
- No VS Code connection
- No SSH connection
- No active dstack attach

## Making Changes

Edit `dev.dstack.yml` to customize:
- GPU memory: `resources.gpu.memory: 24GB`
- Python version: `python: "3.12"`
- Inactivity timeout: `inactivity_duration: 4h`
- Spot policy: `spot_policy: on-demand` (never interrupted)

Apply changes in-place:
```bash
dstack apply -f dev.dstack.yml
```

## Advantages vs Docker Builds

| Docker Build | dstack Dev Environment |
|--------------|------------------------|
| 5-10 min rebuild | Instant code changes |
| Must push to registry | Local repo mounted |
| Build on Mac M1/M2 | Run directly on GPU |
| Multi-arch complexity | Native x86_64 |
| Full rebuild per change | Live development |

## Example Workflow

```bash
# 1. Launch dev environment
dstack apply -f dev.dstack.yml

# 2. Open VS Code with provided link

# 3. Edit code in VS Code (changes are immediate)
# Edit parakeet_service/streaming_server.py

# 4. Test immediately
uv run uvicorn parakeet_service.main:app --reload

# 5. Run tests
uv run pytest tests/test_integration.py -v -k test_streaming

# 6. When done
dstack stop parakeet-dev
```

## Transitioning to Production

Once you're happy with changes in dev environment:

```bash
# 1. Commit changes locally
git add .
git commit -m "Your changes"

# 2. Build Docker image
docker buildx build --platform linux/amd64 -f Dockerfile.gpu \
  -t tomheno/parakeet-streaming-stt:latest --push .

# 3. Deploy service
dstack apply -f .dstack.yml
```
