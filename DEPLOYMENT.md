# Deployment Guide

Simple guide for deploying Parakeet TDT streaming STT with dstack.

## Quick Start

### Development Environment

```bash
# 1. Build dev image (first time only)
docker login -u tomheno
docker buildx build --platform linux/amd64 -f Dockerfile.dev -t tomheno/parakeet-dev:latest --push .

# 2. Create fleet (first time only)
dstack fleet create -f fleet.dstack.yml

# 3. Launch dev environment
dstack apply -f dev.dstack.yml

# 4. Open VS Code (click the link from output)
# Your code is mounted from local, changes are instant!

# 5. In VS Code terminal, start server:
uv run uvicorn parakeet_service.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment

```bash
# 1. Build production image
docker login -u tomheno
docker buildx build --platform linux/amd64 -f Dockerfile.gpu -t tomheno/parakeet-streaming-stt:latest --push .

# 2. Deploy service
dstack apply -f .dstack.yml

# 3. Get endpoint URL
dstack ps
```

## Configuration Files

- **`.dstack.yml`** - Production service configuration
- **`dev-image.dstack.yml`** - Development environment configuration
- **`fleet.dstack.yml`** - Fleet configuration (runpod, single instance)
- **`Dockerfile.gpu`** - Production Docker image
- **`Dockerfile.dev`** - Development Docker image

## Dev Environment Details

**What you get:**
- GPU instance on runpod (eur-is-1)
- VS Code remote connection
- Your code mounted from local (instant changes)
- All dependencies pre-installed
- Auto-stop after 2h inactivity

**Workflow:**
1. Edit code locally → changes sync automatically
2. Test in VS Code terminal on GPU instance
3. No rebuild needed for code changes
4. Only rebuild image when dependencies change

## Production Service Details

**What you get:**
- GPU instance with Parakeet TDT model
- WebSocket streaming endpoint at `:8000/stream`
- Health check at `:8000/healthz`
- Auto-scaling (1-10 replicas based on RPS)

**Testing:**
```bash
# From local machine
python test_microphone.py --url ws://<instance-ip>:8000/stream --language en
```

## Cost Optimization

**Dev environment:**
- Uses spot instances (~70% cheaper)
- Auto-stops after 2h inactivity
- Auto-stops if GPU utilization < 5% for 1h
- Fleet maintains single instance (no auto-scaling)

**Production:**
- On-demand instances for reliability
- Auto-scales 1-10 replicas based on traffic
- Price limit: check `.dstack.yml`

## Common Commands

```bash
# List running environments/services
dstack ps

# Stop dev environment
dstack stop parakeet-dev

# View logs
dstack logs parakeet-dev

# List fleets
dstack fleet list

# Delete fleet
dstack fleet delete parakeet-dev-fleet
```

## Updating Dependencies

```bash
# 1. Edit pyproject.toml locally

# 2. Rebuild dev image
docker buildx build --platform linux/amd64 -f Dockerfile.dev -t tomheno/parakeet-dev:latest --push .

# 3. Restart dev environment
dstack stop parakeet-dev
dstack apply -f dev.dstack.yml
```

## Troubleshooting

**Image build fails:**
- Ensure Docker buildx is set up: `docker buildx create --use --name multiarch`
- Login to Docker Hub: `docker login -u tomheno`

**Dev environment won't start:**
- Check fleet exists: `dstack fleet list`
- Create if missing: `dstack fleet create -f fleet.dstack.yml`

**Can't connect to VS Code:**
- Check status: `dstack ps -v`
- View logs: `dstack logs parakeet-dev --diagnose`

**GPU not available:**
- In VS Code terminal: `nvidia-smi`
- Check CUDA: `uv run python -c 'import torch; print(torch.cuda.is_available())'`

## Architecture

```
Local Machine                Remote GPU (runpod)
├── Edit code            →   ├── Code mounted via repos
├── Git commits          →   ├── Parakeet TDT model loaded
└── Docker builds        →   └── WebSocket streaming server

Dev: Code synced instantly
Prod: Docker image deployed
```

## Next Steps

1. See `README.md` for project overview
2. See `TESTING.md` for running tests
3. Check `pyproject.toml` for dependencies
