#!/bin/bash
# Bash script to build the AIO services Docker image
# Usage: ./build.sh

set -e  # Exit on error

echo "Building AIO Services Docker Image..."

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build the Docker image
IMAGE_NAME="crbot-aio-services"
IMAGE_TAG="latest"

echo ""
echo "Building image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo ""
echo "✓ Docker image built successfully!"
echo ""
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To run the container:"
echo "docker run -d --name aio-services --env-file .env -p 5570:5570 -p 5551:5551 -p 5554:5554 -p 5560:5560 ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "docker tag ${IMAGE_NAME}:${IMAGE_TAG} ghcr.io/deep-jiwan/crbot/aio-services:latest"

docker tag ${IMAGE_NAME}:${IMAGE_TAG} ghcr.io/deep-jiwan/crbot/aio-services:latest
echo ""
echo "ghcr.io/deep-jiwan/crbot/aio-services:latest"