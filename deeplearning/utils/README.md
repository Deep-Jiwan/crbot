# Utils Package

This package contains reusable utility modules for the Clash Royale AI bot.

## Modules

### 1. `health_checker.py`
Health checking functionality for all bot services (ZMQ and HTTP/HTTPS endpoints).

**Features:**
- Check ZMQ services (Publisher, Elixir, Cards, Troops, Win)
- Check HTTP/HTTPS services (Inference API)
- Wait for services to become healthy
- Continuous monitoring mode
- Standalone CLI support

**Usage:**
```python
from utils.health_checker import HealthChecker

checker = HealthChecker()
if checker.are_all_required_services_healthy():
    print("All services ready!")
```

**Standalone:**
```bash
python -m utils.health_checker --check
python -m utils.health_checker --wait 30
python -m utils.health_checker --continuous
```

---

### 2. `data_aggregator.py`
Aggregates data from all bot modules via ZeroMQ and provides a unified interface.

**Features:**
- Subscribes to elixir, cards, troops, and win detection data
- Thread-safe data access
- Real-time data aggregation
- Context manager support

**Usage:**
```python
from utils.data_aggregator import DataAggregator

with DataAggregator() as aggregator:
    data = aggregator.get_current_data()
    print(f"Elixir: {data['elixir']}")
    print(f"Cards: {data['cards']}")
    print(f"Troops: {data['troops']}")
```

---

### 3. `connection_manager.py`
Manages ZeroMQ connections to all bot services with subscription management.

**Features:**
- Centralized connection management
- Service health checking
- Topic subscription management
- Message receiving (blocking/non-blocking)
- Context manager support

**Usage:**
```python
from utils.connection_manager import ConnectionManager

with ConnectionManager() as manager:
    while True:
        msg = manager.receive_message()
        if msg:
            topic, data = msg
            print(f"{topic}: {data}")
```

**Standalone:**
```bash
python -m utils.connection_manager --check
python -m utils.connection_manager --monitor 30
```

---

## Installation

No additional installation required. All modules use standard dependencies from the main `requirements.txt`.

## Configuration

All modules automatically load configuration from the `.env` file in the `deeplearning` directory:

```env
STATE_PUBLISHER_PORT=5550
ELIXIR_PORT=5560
WIN_PORT=5570
TROOPS_PORT=5580
CARDS_PORT=5590
INFERENCE_LINK=http://localhost:9001
```

## Examples

See `health_checker_examples.py` in the parent directory for comprehensive usage examples.

## Import Methods

### Method 1: Import from utils package
```python
from utils import HealthChecker, DataAggregator, ConnectionManager
```

### Method 2: Import specific modules
```python
from utils.health_checker import HealthChecker
from utils.data_aggregator import DataAggregator
from utils.connection_manager import ConnectionManager
```

### Method 3: Import all from module
```python
from utils.health_checker import HealthChecker, ServiceType, HealthCheckResult
```

---

## Module Dependencies

```
utils/
├── __init__.py (exports all classes)
├── health_checker.py (standalone)
├── data_aggregator.py (standalone)
└── connection_manager.py (standalone)
```

All modules are designed to be standalone and can be used independently or together.
