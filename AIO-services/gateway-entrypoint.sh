#!/bin/sh

# Gateway entrypoint script for socat forwarding
# Forwards traffic from container port 5550 to host port 5550

echo "Starting gateway service..."
echo "Forwarding traffic from container:5550 to host.docker.internal:5550"

# Execute socat with proper argument separation
exec socat TCP-LISTEN:5550,fork,reuseaddr TCP-CONNECT:host.docker.internal:5550
