# AIO Services - Architecture Diagram

## Directory Structure
```
AIO-services/
└── services/                        # Service modules
    ├── __init__.py
    │
    ├── winwin/
    │   ├── __init__.py
    │   └── winwin_service.py        # Win/Lose detection
    │
    ├── elixircount/
    │   ├── __init__.py
    │   └── elixir_service.py        # Elixir counting
    │
    ├── carddetection/
    │   ├── __init__.py
    │   └── card_service.py          # Card detection
    │
    └── troopdetection/
        ├── __init__.py
        └── troop_service.py         # Troop detection
```

## Container Runtime Structure
```
┌─────────────────────────────────────────────────────────────┐
│                    AIO Services Container                   │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              main.py (Orchestrator)                   │ │
│  │                                                       │ │
│  │  • Creates shared ZMQ context                        │ │
│  │  • Spawns 4 service threads                          │ │
│  │  • Monitors thread health                            │ │
│  │  • Handles graceful shutdown                         │ │
│  └───────────────────────────────────────────────────────┘ │
│           │           │           │           │             │
│           ▼           ▼           ▼           ▼             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐
│  │  Thread 1   │ │  Thread 2   │ │  Thread 3   │ │ Thread 4 │
│  │             │ │             │ │             │ │          │
│  │   WinWin    │ │   Elixir    │ │   Cards     │ │  Troops  │
│  │  Service    │ │  Service    │ │  Service    │ │ Service  │
│  │             │ │             │ │             │ │          │
│  │ Port: 5570  │ │ Port: 5551  │ │ Port: 5554  │ │Port: 5560│
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘
│           │           │           │           │             │
└───────────┼───────────┼───────────┼───────────┼─────────────┘
            │           │           │           │
            ▼           ▼           ▼           ▼
       ZMQ PUB     ZMQ PUB     ZMQ PUB     ZMQ PUB
    topic:winner topic:ecount topic:cards topic:troops
```

## Data Flow Diagram
```
                    ┌─────────────────┐
                    │  Frame Source   │
                    │  (Publisher)    │
                    │  Port: 5550     │
                    └────────┬────────┘
                             │
                    topic: frame|<jpg>
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌──────────────┐
│ Frame arrives │    │ Frame arrives │    │Frame arrives │
│ at WinWin     │    │ at Elixir     │    │at Cards/     │
│               │    │               │    │Troops        │
│ 1. Decode JPG │    │ 1. Decode JPG │    │1. Decode JPG │
│ 2. Resize     │    │ 2. Resize     │    │2. Resize     │
│ 3. Process    │    │ 3. Count      │    │3. Detect/    │
│ 4. Detect win │    │    elixir     │    │   Classify   │
│               │    │               │    │              │
└───────┬───────┘    └───────┬───────┘    └──────┬───────┘
        │                    │                    │
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌──────────────┐
│  Publish to   │    │  Publish to   │    │ Publish to   │
│  Port 5570    │    │  Port 5551    │    │Ports 5554/60 │
│               │    │               │    │              │
│winner|True    │    │ecount|5       │    │cards|1:knight│
│winner|False   │    │ecount|7       │    │troops|{json} │
│winner|ongoing │    │ecount|10      │    │              │
└───────────────┘    └───────────────┘    └──────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Subscribers   │
                    │ (Master Receiver│
                    │  or other apps) │
                    └─────────────────┘
```

## Service Details

### 1. WinWin Detection Service (Port 5570)
```
Input:  frame|<jpg_bytes>
Process: 
  • Sample pixels at 4 specific coordinates
  • Match against 2 color targets
  • Determine win/lose/ongoing state
Output: winner|True/False/ongoing
```

### 2. Elixir Counter Service (Port 5560)
```
Input:  frame|<jpg_bytes>
Process:
  • Sample pixels along elixir bar
  • Count matches to purple color
  • Calculate elixir count (0-10)
Output: ecount|0-10
```

### 3. Card Detection Service (Port 5590)
```
Input:  frame|<jpg_bytes>
Process:
  • Extract 4 card slots from frame
  • Send to Roboflow for classification
  • Map results to slot numbers
Output: cards|1:card1,2:card2,3:card3,4:card4
```

### 4. Troop Detection Service (Port 5580)
```
Input:  frame|<jpg_bytes>
Process:
  • Send full frame to Roboflow workflow
  • Receive detection results (troops, positions)
  • Clean and format JSON
Output: troops|{json_detection_data}
```

## Network Configuration

### Docker Network
```
┌──────────────────────────────────────────────────┐
│              cr_network (bridge)                 │
│                                                  │
│  ┌──────────────┐        ┌──────────────────┐  │
│  │   Gateway    │        │  AIO Services    │  │
│  │              │        │                  │  │
│  │ Internal:    │ ZMQ    │ Subscribes to:   │  │
│  │ 5550         │───────▶│ gateway:5550     │  │
│  │              │        │                  │  │
│  │ External:    │        │ Publishes on:    │  │
│  │ host:5550    │        │ • 5570 (WinWin)  │  │
│  └──────────────┘        │ • 5551 (Elixir)  │  │
│                          │ • 5554 (Cards)   │  │
│                          │ • 5560 (Troops)  │  │
│                          └──────────────────┘  │
└──────────────────────────────────────────────────┘
         ▲                          │
         │                          │
    Host Publisher          Exposed Ports
    (localhost:5550)        (5570,5551,5554,5560)
```

## Build Process

### Dockerfile Steps
```
1. FROM python:3.11-slim
   └─ Base image with Python

2. Install system dependencies
   ├─ libglib2.0-0
   ├─ libgl1
   ├─ libsm6
   ├─ libxext6
   └─ libxrender-dev

3. WORKDIR /app
   └─ Set working directory

4. COPY requirements.txt
   └─ Copy dependency list

5. RUN pip install
   └─ Install all Python packages

6. COPY services/ & main.py
   └─ Copy application code

7. Create runtime directories
   ├─ services/winwin/images
   ├─ services/winwin/frame
   ├─ services/elixircount/images
   ├─ services/elixircount/frame
   ├─ services/carddetection/images
   ├─ services/carddetection/cards
   └─ services/troopdetection/temp

8. EXPOSE ports

9. CMD ["python", "main.py"]
   └─ Start orchestrator
```

## Environment Configuration Flow
```
.env file
    │
    ├─ FRAME_WIDTH ────────────┐
    ├─ FRAME_HEIGHT ───────────┤
    ├─ ZMQ_ADDRESS ────────────┤─▶ All Services
    ├─ SLEEP_TIME ─────────────┤
    └─ ANNOTATE ───────────────┘
    │
    ├─ WINNER_Y1 ──────────────┐
    ├─ WINNER_Y2 ──────────────┤
    ├─ TARGET_B1 ──────────────┤─▶ WinWin Service
    ├─ TARGET_G1 ──────────────┤
    └─ ...─────────────────────┘
    │
    ├─ ELIXIR_Y ───────────────┐
    ├─ ELIXIR_X_START ─────────┤─▶ Elixir Service
    ├─ TARGET_B ───────────────┤
    └─ ...─────────────────────┘
    │
    ├─ ROBOFLOW_URL ───────────┐
    ├─ ROBOFLOW_API_KEY ───────┤─▶ Card & Troop
    ├─ WORKSPACE_NAME ─────────┤   Services
    └─ WORKFLOW_ID ────────────┘
```

## Thread Lifecycle
```
main.py starts
    │
    ├─ Create ZMQ Context (shared)
    │
    ├─ Spawn Thread 1: WinWin Service
    │   └─ while True: receive frame → process → publish
    │
    ├─ Spawn Thread 2: Elixir Service
    │   └─ while True: receive frame → process → publish
    │
    ├─ Spawn Thread 3: Card Service
    │   └─ while True: receive frame → process → publish
    │
    ├─ Spawn Thread 4: Troop Service
    │   └─ while True: receive frame → process → publish
    │
    └─ Monitor Loop
        ├─ Check thread health every 5s
        ├─ Log warnings if thread dies
        └─ Wait for Ctrl+C → cleanup
```