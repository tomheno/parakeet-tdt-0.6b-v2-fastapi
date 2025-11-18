# dstack Development Workflows

Two approaches for working with dstack dev environments.

## Approach 1: Python + NVCC (No Docker Build)

**Best for**: Quick experimentation, no Docker complexity

**File**: `dev.dstack.yml`

### Advantages
- No Docker build required
- Start immediately with `dstack apply -f dev.dstack.yml`
- Dependencies installed on first run

### Disadvantages
- Slower first startup (~5-10 min for dependency installation)
- Installation happens every time you create a new environment

### Usage
```bash
dstack apply -f dev.dstack.yml
```

---

## Approach 2: Pre-built Docker Image (Recommended)

**Best for**: Regular development, faster startup, reproducible environments

**Files**: `Dockerfile.dev`, `dev-image.dstack.yml`, `fleet.dstack.yml`

### Advantages
- Fast startup (~1-2 min, dependencies pre-installed)
- Reproducible environment
- Can be shared with team
- Fleet management for cost optimization

### Disadvantages
- Initial Docker build required (~10 min)
- Must rebuild to update dependencies

---

## Complete Workflow: Docker Image Approach

### 1. Create Fleet (One-Time Setup)

```bash
dstack fleet create -f fleet.dstack.yml
```

This creates a managed pool of GPU instances with:
- Auto-scaling (0-2 instances)
- Cost optimization (runpod, vastai, lambda backends)
- 1 hour idle timeout

### 2. Build and Push Dev Image

#### Set up authentication
```bash
# Login to Docker Hub
docker login -u tomheno

# Or use environment variable
export DOCKER_PASSWORD="your_docker_hub_token"
echo $DOCKER_PASSWORD | docker login -u tomheno --password-stdin
```

#### Build and push
```bash
# Make script executable
chmod +x build-dev-image.sh

# Build and push
./build-dev-image.sh latest

# Or build manually
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.dev \
  -t tomheno/parakeet-dev:latest \
  --push .
```

### 3. Launch Dev Environment

```bash
dstack apply -f dev-image.dstack.yml
```

Select an offer, then open VS Code with the provided link.

### 4. Development in VS Code

```bash
# Server runs on GPU instance, code synced from local
# Changes to code are instant (mounted from local)

# Start server with auto-reload
uv run uvicorn parakeet_service.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
uv run pytest tests/test_sanity.py -v
uv run pytest tests/test_integration.py -v

# Test from local machine
python test_microphone.py --url ws://<instance-ip>:8000/stream
```

### 5. Update Dependencies (When Needed)

If you add new dependencies to `pyproject.toml`:

```bash
# 1. Rebuild image
./build-dev-image.sh latest

# 2. Restart dev environment
dstack stop parakeet-dev
dstack apply -f dev-image.dstack.yml
```

---

## Configuration Details

### Fleet Configuration (`fleet.dstack.yml`)

```yaml
nodes: 0..2              # Auto-scale 0-2 instances
idle_duration: 1h        # Terminate idle instances after 1 hour
backends: [runpod, vastai, lambda]  # Cost-effective GPU providers
```

### Dev Environment (`dev-image.dstack.yml`)

**Resource Management:**
- `max_price: 1.00` - Maximum $1/hour per instance
- `spot_policy: auto` - Use spot instances (~70% cheaper)
- `inactivity_duration: 2h` - Auto-stop after 2 hours inactive

**Auto-Termination:**
- `utilization_policy`: Stops if GPU utilization < 5% for 1 hour
- Saves costs on forgotten instances

**Volumes (Optional):**
```yaml
volumes:
  - name: parakeet-models
    path: /models
```

Create volume first:
```bash
dstack volume create parakeet-models --size 100GB
```

---

## Comparison

| Feature | Python + NVCC | Docker Image |
|---------|---------------|--------------|
| **First startup** | 5-10 min | 1-2 min |
| **Subsequent startups** | 5-10 min | 1-2 min |
| **Initial setup** | None | 10 min build |
| **Dependency updates** | Automatic | Rebuild required |
| **Reproducibility** | Medium | High |
| **Team sharing** | Harder | Easier |
| **Best for** | Quick experiments | Regular development |

---

## Management Commands

```bash
# List fleets
dstack fleet list

# List dev environments
dstack ps
dstack ps -v  # Verbose (shows inactivity time)

# Monitor live
dstack ps --watch

# Attach to running environment
dstack attach parakeet-dev

# Stop environment
dstack stop parakeet-dev

# View logs
dstack logs parakeet-dev
dstack logs parakeet-dev --diagnose  # For debugging failures

# Delete fleet
dstack fleet delete parakeet-dev-fleet
```

---

## Cost Optimization Tips

1. **Use spot instances**: Set `spot_policy: auto` (~70% savings)
2. **Set max_price**: Prevent expensive instances
3. **Use fleets**: Automatic idle instance termination
4. **Set inactivity_duration**: Auto-stop forgotten environments
5. **Use utilization_policy**: Auto-stop underutilized instances
6. **Choose right backends**: runpod/vastai typically cheaper than AWS/GCP

---

## Troubleshooting

### Image pull fails (private registry)

Add registry authentication:
```yaml
registry_auth:
  username: tomheno
  password: ${{ env.GITHUB_TOKEN }}
```

Or make image public on GitHub Container Registry.

### Dependencies out of sync

Rebuild image after updating `pyproject.toml`:
```bash
./build-dev-image.sh latest
```

### Fleet has no instances

Check fleet status:
```bash
dstack fleet list
```

If fleet shows 0 instances, it's normal - instances are created on-demand.

### GPU not detected

Check in VS Code terminal:
```bash
nvidia-smi
uv run python -c 'import torch; print(torch.cuda.is_available())'
```

---

## Example: Full Development Session

```bash
# 1. Create fleet (first time only)
dstack fleet create -f fleet.dstack.yml

# 2. Build dev image (first time or after dependency changes)
export GITHUB_TOKEN="ghp_your_token"
./build-dev-image.sh latest

# 3. Launch dev environment
dstack apply -f dev-image.dstack.yml
# Select offer, wait 1-2 minutes

# 4. Open VS Code (click link from output)

# 5. Develop (in VS Code terminal on GPU instance)
uv run uvicorn parakeet_service.main:app --reload --host 0.0.0.0 --port 8000

# 6. Test from local machine
python test_microphone.py --url ws://<instance-ip>:8000/stream --language en

# 7. Make changes locally (synced automatically)
# Edit parakeet_service/streaming_server.py
# Server auto-reloads

# 8. Run tests
uv run pytest tests/test_integration.py -v

# 9. Done - stop environment
dstack stop parakeet-dev
# Fleet keeps instance alive for 1 hour (idle_duration)
# Instance terminates after 1 hour if not reused
```

---

## Transitioning to Production

Once development is complete:

```bash
# 1. Commit changes
git add .
git commit -m "Implement feature X"
git push

# 2. Build production image
docker buildx build --platform linux/amd64 -f Dockerfile.gpu \
  -t tomheno/parakeet-streaming-stt:latest --push .

# 3. Deploy service
dstack apply -f .dstack.yml
```
