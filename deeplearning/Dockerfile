# Enhanced Clash Royale Deep Learning AI Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs data checkpoints

# Create non-root user for security
RUN useradd -m -u 1000 aiuser && \
    chown -R aiuser:aiuser /app
USER aiuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import cv2; import zmq; print('Health check passed')" || exit 1

# Default command
CMD ["python", "main.py", "--mode", "play"]

# Expose ports for ZeroMQ communication
EXPOSE 5550 5551 5552 5560 5570 5580

# Labels for metadata
LABEL maintainer="Clash Royale AI Team"
LABEL version="2.0.0"
LABEL description="Enhanced Clash Royale Deep Learning AI with 15 features, zone prediction, and JSON output"
LABEL features="Enhanced AI, Zone Prediction, JSON Output, Comprehensive Logging"
