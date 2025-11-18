#!/bin/bash
set -e

# Configuration
IMAGE_NAME="tomheno/parakeet-dev"
TAG="${1:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"

echo "Building development image: ${FULL_IMAGE}"
echo "============================================"

# Login to Docker Hub
echo "Logging in to Docker Hub..."
if [ -z "$DOCKER_PASSWORD" ]; then
    echo "Warning: DOCKER_PASSWORD not set. You may need to login manually."
    echo "Run: docker login -u tomheno"
else
    echo "$DOCKER_PASSWORD" | docker login -u tomheno --password-stdin
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
echo "Get the SHA256 digest:"
echo "docker inspect ${FULL_IMAGE} --format='{{index .RepoDigests 0}}'"
echo ""
echo "Next steps:"
echo "1. Update dev-image.dstack.yml with the image: ${FULL_IMAGE}"
echo "2. Create fleet: dstack fleet create -f fleet.dstack.yml"
echo "3. Launch dev environment: dstack apply -f dev-image.dstack.yml"
