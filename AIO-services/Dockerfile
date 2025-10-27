# Use Python 3.11 slim base image
FROM python:3.11-slim

# Install OpenCV and system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all service code
COPY services/ ./services/
COPY main.py .

# Create directories for runtime data
RUN mkdir -p services/winwin/images services/winwin/frame \
             services/elixircount/images services/elixircount/frame \
             services/carddetection/images services/carddetection/cards \
             services/troopdetection/temp

# Expose all service ports
# WINWIN_PUB_PORT=5570
# ELIXIR_PUB_PORT=5560
# CARD_PUB_PORT=5590
# TROOP_PUB_PORT=5580
EXPOSE 5570 5560 5590 5580

# Run the main orchestrator
CMD ["python", "main.py"]
