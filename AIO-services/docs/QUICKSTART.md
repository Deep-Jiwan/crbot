# AIO Services - Build and Deployment Guide

## Quick Start

### 1. Prepare Environment Configuration

```bash
cd AIO-services
cp .env.example .env
```

Edit `.env` with your configuration:
- Set `ZMQ_ADDRESS` for frame subscription
- Add Roboflow API credentials
- Adjust detection parameters as needed

### 2. Build the Docker Image

**On Windows (PowerShell):**
```powershell
.\build.ps1
```

**On Linux/Mac:**
```bash
chmod +x build.sh
./build.sh
```

**Or manually:**
```bash
docker build -t crbot-aio-services:latest .
```

### 3. Run the Container

**Using docker run:**
```bash
docker run -d \
  --name aio-services \
  --env-file .env \
  -p 5570:5570 \
  -p 5551:5551 \
  -p 5554:5554 \
  -p 5560:5560 \
  crbot-aio-services:latest
```

**Using docker-compose:**
```bash
# Use the provided docker-compose-example.yaml as reference
docker-compose up -d
# make sure to use the correct image and if you have built it, verify you are using the correct tagged image
```


### 4. Test the Services

```bash
# Install pyzmq if not already installed
pip install pyzmq

# Run the test script
python test_services.py

# Or test against a remote host
python test_services.py 192.168.1.100
```

### 5. View Logs

```bash
# Follow all service logs
docker logs -f aio-services

# Search for specific service
docker logs aio-services | grep "\[WINWIN\]"
docker logs aio-services | grep "\[ELIXIR\]"
docker logs aio-services | grep "\[CARDS\]"
docker logs aio-services | grep "\[TROOPS\]"
```

## Port Mapping Reference

| Service | Internal Port | External Port | Topic | Data Format |
|---------|---------------|---------------|-------|-------------|
| WinWin Detection | 5570 | 5570 | `winner\|` | `winner\|True/False/ongoing` |
| Elixir Counter | 5560 | 5560 | `ecount\|` | `ecount\|0-10` |
| Card Detection | 5590 | 5590 | `cards\|` | `cards\|1:card1,2:card2,...` |
| Troop Detection | 5580 | 5580 | `troops\|` | `troops\|{json_data}` |

## Environment Variables

### Required for All Services
- `ZMQ_ADDRESS` - Frame publisher address (e.g., `tcp://gateway:5550`)
- `FRAME_WIDTH` - Frame width in pixels (default: 1080)
- `FRAME_HEIGHT` - Frame height in pixels (default: 1920)

###  Publishing Ports
Each service has a default publishing port, but you can override them:
- `WINWIN_PUB_PORT` - WinWin service port (default: 5570)
- `ELIXIR_PUB_PORT` - Elixir service port (default: 5560)
- `CARD_PUB_PORT` - Card service port (default: 5590)
- `TROOP_PUB_PORT` - Troop service port (default: 5580)

### Required for Card Detection
- `CARD_DETECTION_ROBOFLOW_API_KEY` - Your Roboflow API key
- `CARD_DETECTION_ROBOFLOW_URL` - Roboflow inference server URL
- `CARD_DETECTION_WORKSPACE_NAME` - Roboflow workspace name
- `CARD_DETECTION_WORKFLOW_ID` - Roboflow workflow ID (e.g., custom-workflow)

### Required for Troop Detection
- `TROOP_DETECTION_ROBOFLOW_API_KEY` - Your Roboflow API key
- `TROOP_DETECTION_ROBOFLOW_URL` - Roboflow inference server URL
- `TROOP_DETECTION_WORKSPACE_NAME` - Roboflow workspace name
- `TROOP_DETECTION_WORKFLOW_ID` - Roboflow workflow ID (e.g., detect-count-and-visualize)

### Optional
- `SLEEP_TIME` - Delay between frame processing (default: 0.1)
- `ANNOTATE` - Enable/disable debug annotations (default: True)

## Troubleshooting

### Container won't start
```bash
# Check container logs
docker logs aio-services

# Check for port conflicts
docker ps | grep "5560\|5570\|5580\|5590"
```

### Service not responding
```bash
# Check if service started
docker logs aio-services | grep "Service started"

# Check for errors
docker logs aio-services | grep "ERROR"
```

### Environment variable issues
```bash
# List all environment variables in container
docker exec aio-services env

# Check specific variable
docker exec aio-services env | grep ROBOFLOW_API_KEY
```

### Roboflow connection errors
```bash
# Test Roboflow URL from container
docker exec aio-services ping -c 3 roboinference

# Check API keys are set
docker exec aio-services env | grep "DETECTION_ROBOFLOW"
```

## Publishing to Registry

### GitHub Container Registry (GHCR)

```bash
# Tag the image
docker tag crbot-aio-services:latest ghcr.io/deep-jiwan/crbot/aio-services:latest

# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u deep-jiwan --password-stdin

# Push to registry
docker push ghcr.io/deep-jiwan/crbot/aio-services:latest
```

### Docker Hub

```bash
# Tag the image
docker tag crbot-aio-services:latest your-username/crbot-aio-services:latest

# Login to Docker Hub
docker login

# Push to registry
docker push your-username/crbot-aio-services:latest
```

## Updating Services

When you need to update the services:

1. Stop the running container:
   ```bash
   docker stop aio-services
   docker rm aio-services
   ```

2. Pull the latest image (if using registry):
   ```bash
   docker pull ghcr.io/deep-jiwan/crbot/aio-services:latest
   ```

3. Or rebuild locally:
   ```bash
   docker build -t crbot-aio-services:latest .
   ```

4. Start the new container:
   ```bash
   docker run -d --name aio-services --env-file .env -p 5570:5570 -p 5551:5551 -p 5554:5554 -p 5560:5560 crbot-aio-services:latest
   ```


## Support

For issues or questions:
1. Check logs: `docker logs aio-services`
2. Run test script: `python test_services.py`
3. Verify environment variables are set correctly
4. Check network connectivity to ZMQ publisher
5. For Roboflow issues, verify API credentials
