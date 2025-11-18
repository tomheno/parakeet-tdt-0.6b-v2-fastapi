#!/bin/bash
set -e

# Configuration
IMAGE_NAME="ghcr.io/tomheno/parakeet-dev"
TAG="${1:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"

echo "Building development image: ${FULL_IMAGE}"
echo "============================================"

# Login to GitHub Container Registry
echo "Logging in to GitHub Container Registry..."
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Warning: GITHUB_TOKEN not set. You may need to login manually."
    echo "Run: echo \$GITHUB_TOKEN | docker login ghcr.io -u tomheno --password-stdin"
else
    echo "$GITHUB_TOKEN" | docker login ghcr.io -u tomheno --password-stdin
fi

# Build for linux/amd64 (GPU servers)
echo ""
echo "Building image for linux/amd64..."
docker buildx build \
    --platform linux/amd64 \
    -f Dockerfile.dev \
    -t "${FULL_IMAGE}" \
    --push \
    .

echo ""
echo "============================================"
echo "✅ Image built and pushed: ${FULL_IMAGE}"
echo ""
echo "Next steps:"
echo "1. Update dev-image.dstack.yml with the image: ${FULL_IMAGE}"
echo "2. Create fleet: dstack fleet create -f fleet.dstack.yml"
echo "3. Launch dev environment: dstack apply -f dev-image.dstack.yml"
