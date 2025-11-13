# Deployment Guide

Deploy Parakeet TDT Streaming STT on GPU-enabled cloud platforms.

## Quick Deploy Options

### 1. dstack (Recommended)

dstack provides multi-cloud GPU deployment with auto-scaling.

#### Prerequisites

```bash
uv tool install dstack
dstack server
```

#### Deploy

```bash
# Initialize dstack project
dstack init

# Deploy service
dstack run .

# Or specify configuration explicitly
dstack apply -f .dstack.yml
```

#### Check status

```bash
dstack ps
```

#### Configuration

Edit `.dstack.yml` to customize:
- GPU requirements (A10G, L4, T4, etc.)
- Memory limits
- Auto-scaling rules
- Environment variables

### 2. RunPod

#### Using Docker

1. Create RunPod account at https://runpod.io
2. Deploy pod with NVIDIA GPU (A4000, A5000, or better)
3. Use Docker deployment:

```bash
# Build and push to registry
docker build -f Dockerfile.gpu -t your-registry/parakeet-stt:latest .
docker push your-registry/parakeet-stt:latest

# On RunPod:
# - Select GPU pod
# - Use custom Docker image: your-registry/parakeet-stt:latest
# - Expose port 8000
# - Start pod
```

#### Using Template

Create RunPod template:

```yaml
Name: Parakeet Streaming STT
Docker Image: nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
Container Disk: 50 GB
Expose HTTP Ports: 8000
Docker Command: |
  apt-get update && apt-get install -y git python3-pip &&
  git clone <your-repo-url> /app &&
  cd /app &&
  pip3 install -r requirements.txt &&
  uvicorn parakeet_service.main:app --host 0.0.0.0 --port 8000
```

### 3. AWS EC2 (G4/G5 instances)

#### Launch instance

```bash
# Use AWS CLI or console
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \  # Deep Learning AMI
  --instance-type g4dn.xlarge \
  --key-name your-key \
  --security-groups gpu-sg \
  --user-data file://deploy.sh
```

#### deploy.sh

```bash
#!/bin/bash
cd /home/ubuntu
git clone <your-repo-url> parakeet-stt
cd parakeet-stt

# Build and run
docker build -f Dockerfile.gpu -t parakeet-stt .
docker run -d --gpus all -p 8000:8000 parakeet-stt
```

### 4. GCP (with NVIDIA GPUs)

```bash
gcloud compute instances create parakeet-stt \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --metadata-from-file startup-script=deploy.sh
```

## Local Testing

### Build GPU image

```bash
docker build -f Dockerfile.gpu -t parakeet-stt:gpu .
```

### Run with NVIDIA runtime

```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -e MODEL_PRECISION=fp16 \
  -e DEVICE=cuda \
  --name parakeet-stt \
  parakeet-stt:gpu
```

### Test

```bash
# Health check
curl http://localhost:8000/healthz

# WebSocket test
python test_streaming_client.py test.wav
```

## Environment Variables

Configure deployment via environment variables:

```bash
MODEL_PRECISION=fp16     # fp16 or fp32 (fp16 recommended for speed)
DEVICE=cuda              # cuda or cpu
BATCH_SIZE=4             # Batch size for REST API
LOG_LEVEL=INFO           # DEBUG, INFO, WARNING, ERROR
TARGET_SR=16000          # Audio sample rate
MAX_AUDIO_DURATION=30    # Max audio length (seconds)
```

## GPU Requirements

### Minimum

- GPU: NVIDIA T4 (16GB VRAM)
- CPU: 4 cores
- RAM: 16GB
- Disk: 50GB

### Recommended

- GPU: NVIDIA A10G (24GB VRAM)
- CPU: 8 cores
- RAM: 32GB
- Disk: 100GB
- Network: 10 Gbps

### Capacity

Per A10G (24GB):
- Concurrent streams: ~20
- Throughput: ~2000 minutes/hour
- Latency: <300ms

## Monitoring

### Health checks

```bash
# Basic health
curl http://your-deployment:8000/healthz

# Model info
curl http://your-deployment:8000/config
```

### Logs

```bash
# Docker
docker logs -f parakeet-stt

# dstack
dstack logs <run-name>
```

### Metrics

Monitor:
- GPU utilization (target: 60-80%)
- Memory usage (should be <90%)
- Latency (target: <300ms)
- Error rate (target: <1%)

## Scaling

### Horizontal scaling (dstack)

```yaml
# In .dstack.yml
replicas: 1..10
autoscaling:
  metric: gpu_utilization
  target: 70
```

### Load balancing

Use NGINX or cloud load balancer:

```nginx
upstream parakeet {
    least_conn;
    server instance1:8000;
    server instance2:8000;
    server instance3:8000;
}

server {
    listen 80;

    location / {
        proxy_pass http://parakeet;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Cost Optimization

### GPU selection by use case

| Use Case | GPU | Cost/hr | Streams | $/stream/hr |
|----------|-----|---------|---------|-------------|
| Development | T4 | $0.35 | 10 | $0.035 |
| Production | A10G | $1.00 | 20 | $0.05 |
| High-scale | A100 | $3.00 | 50 | $0.06 |

### Tips

1. Use FP16 precision (2x speedup, same accuracy)
2. Enable model caching in dstack
3. Use spot instances (70% cheaper)
4. Auto-scale during off-peak hours
5. Batch REST API requests

## Troubleshooting

### Out of memory

```bash
# Reduce batch size
ENV BATCH_SIZE=2

# Reduce concurrent streams (add to server)
MAX_CONCURRENT_STREAMS=10
```

### Slow inference

```bash
# Check GPU utilization
nvidia-smi

# Verify FP16
docker exec parakeet-stt python -c "import torch; print(next(model.parameters()).dtype)"

# Check CUDA
docker exec parakeet-stt python -c "import torch; print(torch.cuda.is_available())"
```

### Connection issues

```bash
# Check port binding
netstat -tuln | grep 8000

# Check firewall
sudo ufw allow 8000

# Test WebSocket
wscat -c ws://localhost:8000/stream
```

## Security

### Production checklist

- [ ] Use HTTPS/WSS (not HTTP/WS)
- [ ] Add authentication middleware
- [ ] Rate limiting
- [ ] Input validation
- [ ] Network isolation
- [ ] Secrets management
- [ ] Audit logging

### HTTPS with Let's Encrypt

```bash
# Install certbot
apt-get install certbot python3-certbot-nginx

# Get certificate
certbot --nginx -d your-domain.com

# Auto-renewal
systemctl enable certbot.timer
```

## CI/CD

### GitHub Actions example

```yaml
name: Deploy to dstack

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dstack
        run: uv tool install dstack

      - name: Deploy
        env:
          DSTACK_TOKEN: ${{ secrets.DSTACK_TOKEN }}
        run: |
          dstack config --token $DSTACK_TOKEN
          dstack run .
```

## Support

- Documentation: See STREAMING.md
- Issues: GitHub Issues
- Discord: dstack.ai community
