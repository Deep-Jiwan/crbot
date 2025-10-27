# New Integration Layer Guide

The `new_integration_layer.py` is a complete, production-ready integration layer that combines health checking, connection management, data aggregation, and game control.

## Features

✅ **Health Checking** - Verifies all required services are running  
✅ **Connection Management** - Manages ZMQ connections to all services  
✅ **Data Aggregation** - Aggregates data from all bot modules in real-time  
✅ **Game Player Integration** - Controls the game via mouse/keyboard  
✅ **Multiple Modes** - Test, Monitor, Play, Status  
✅ **Graceful Shutdown** - Proper cleanup on exit  

---

## Quick Start

### 1. Check if services are running
```bash
python new_integration_layer.py --status
```

### 2. Run with service checks (recommended)
```bash
python new_integration_layer.py
```

### 3. Skip service checks (if services already running)
```bash
python new_integration_layer.py --no-checks
```

---

## Command Line Options

| Option | Description |
|--------|-------------|
| `--no-checks` | Skip service health checks (assumes services running) |
| `--test` | Run GamePlayer functionality test |
| `--monitor N` | Monitor data for N seconds instead of game loop |
| `--status` | Show service status and exit |

---

## Usage Examples

### Example 1: Check Status
```bash
python new_integration_layer.py --status
```
Output:
```
Checking status...

Services Healthy: True
Connected: False

Service Details:
  ✓ publisher: 15.2ms
  ✓ elixir: 12.8ms
  ✓ cards: 18.3ms
  ✓ troops: 14.1ms
  ✓ win: 11.9ms
```

### Example 2: Test GamePlayer
```bash
python new_integration_layer.py --test
```

### Example 3: Monitor Data
```bash
python new_integration_layer.py --monitor 30
```
Monitors game data for 30 seconds without playing.

### Example 4: Run Game Loop
```bash
python new_integration_layer.py
```
Runs the main AI game loop (add your logic in `run_game_loop()`).

---

## Architecture

```
ClashRoyaleAI
├── HealthChecker (utils.health_checker)
│   └── Checks service availability
├── ConnectionManager (utils.connection_manager)
│   └── Manages ZMQ connections
├── DataAggregator (utils.data_aggregator)
│   └── Aggregates game data
└── GamePlayer (gameplayer)
    └── Controls game actions
```

---

## Accessing Game State

The `ClashRoyaleAI` class provides easy access to game state:

```python
# Get current game state
state = ai.get_game_state()

# Access data
elixir = state['elixir']           # Current elixir count (0-10)
cards = state['cards']             # Available cards
troops = state['troops']           # Troops on battlefield
win_condition = state['win_condition']  # 'ongoing', 'win', 'loss', 'draw'
timestamp = state['timestamp']     # Last update time
```

### Game State Structure

```python
{
    "elixir": 5,                    # int: 0-10
    "cards": [                       # List of available cards
        {
            "slot": 0,               # Card slot (0-3)
            "name": "Knight"         # Card name
        },
        ...
    ],
    "troops": [                      # List of troops on field
        {
            "type": "Knight",        # Troop type
            "confidence": 0.95,      # Detection confidence
            "x": 1600.0,            # X position
            "y": 500.0,             # Y position
            "team": "ally"          # "ally" or "enemy"
        },
        ...
    ],
    "win_condition": "ongoing",      # Game status
    "timestamp": 1234567890.123      # Last update time
}
```

---

## Playing Cards

```python
# In run_game_loop() method:

# Get current state
state = self.get_game_state()

# Check if we have enough elixir
if state['elixir'] >= 5:
    # Play card at slot 2 in center arena
    self.play_card(
        card_index=2,    # Card slot (0-3)
        x=1600,          # X coordinate
        y=500            # Y coordinate
    )
```

---

## Implementing Your AI Logic

Edit the `run_game_loop()` method in `ClashRoyaleAI` class:

```python
def run_game_loop(self):
    """Main game loop - Add your AI logic here"""
    
    print("Starting game loop...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while self.running:
            # Get current game state
            state = self.get_game_state()
            
            # YOUR AI LOGIC HERE
            # Example: Simple elixir-based strategy
            if state['elixir'] >= 7:
                # We have lots of elixir, play aggressively
                if state['cards']:
                    card = state['cards'][0]
                    self.play_card(card['slot'], 1600, 400)
            
            elif state['elixir'] >= 4:
                # Check if enemy troops approaching
                enemy_troops = [t for t in state['troops'] if t['team'] == 'enemy']
                if enemy_troops:
                    # Play defensive card
                    if len(state['cards']) > 1:
                        self.play_card(state['cards'][1]['slot'], 1600, 700)
            
            # Check win condition
            if state['win_condition'] != 'ongoing':
                print(f"\nGame ended: {state['win_condition']}")
                break
            
            # Sleep to prevent busy loop
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nStopping game loop...")
```

---

## Service Requirements

The integration layer requires these services to be running:

### Required Services
- **Publisher** (port 5550) - Publishes game frames
- **Elixir Counter** (port 5560) - Detects elixir count
- **Win Detection** (port 5570) - Detects win/loss/draw

### Optional Services
- **Card Detection** (port 5590) - Detects available cards
- **Troop Detection** (port 5580) - Detects troops on battlefield

---

## Error Handling

The integration layer handles errors gracefully:

```python
try:
    ai = ClashRoyaleAI()
    
    # Initialize (checks services, connects, starts aggregation)
    if not ai.initialize():
        print("Initialization failed")
        return
    
    # Run your logic
    ai.run_game_loop()
    
except KeyboardInterrupt:
    print("Interrupted by user")
    
except Exception as e:
    print(f"Error: {e}")
    
finally:
    # Always cleanup
    ai.cleanup()
```

---

## Advanced Usage

### Custom Initialization

```python
ai = ClashRoyaleAI()

# Initialize with custom timeout
ai.initialize(
    skip_checks=False,     # Don't skip health checks
    wait_timeout=60        # Wait up to 60 seconds for services
)
```

### Direct Component Access

```python
ai = ClashRoyaleAI()

# Access health checker
ai.health_checker.check_service("elixir")

# Access connection manager
ai.connection_manager.subscribe_to_topic("troops")

# Access data aggregator
elixir = ai.data_aggregator.get_elixir()
cards = ai.data_aggregator.get_cards()

# Access game player
ai.game_player.click_card(2)
```

---

## Troubleshooting

### Services not detected
```bash
# Check service status
python new_integration_layer.py --status

# Check with health checker
python -m utils.health_checker
```

### Connection issues
```bash
# Test connection manager
python -m utils.connection_manager --check
```

### Data not updating
```bash
# Monitor data
python new_integration_layer.py --monitor 10
```

---

## Files Structure

```
deeplearning/
├── new_integration_layer.py    ← Main integration file
├── utils/
│   ├── health_checker.py       ← Service health checking
│   ├── connection_manager.py   ← ZMQ connection management
│   ├── data_aggregator.py      ← Data aggregation
│   └── __init__.py             ← Package exports
├── gameplayer/
│   └── gameplayer.py           ← Game control
└── .env                        ← Configuration
```

---

## Next Steps

1. **Test the integration:**
   ```bash
   python new_integration_layer.py --test
   ```

2. **Monitor data flow:**
   ```bash
   python new_integration_layer.py --monitor 30
   ```

3. **Implement your AI logic** in the `run_game_loop()` method

4. **Run your AI:**
   ```bash
   python new_integration_layer.py
   ```

---

## Support

For issues or questions:
- Check service status: `python new_integration_layer.py --status`
- View utils documentation: `deeplearning/utils/README.md`
- Test individual components in `utils/` folder
