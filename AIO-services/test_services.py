#!/usr/bin/env python3
"""
Test script to validate AIO services container
Connects to all 4 service ports and prints received messages
"""
import zmq
import time
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env.aio
load_dotenv('.env.aio')

# Read ports from environment with defaults
SERVICES = {
    "WinWin": {"port": int(os.getenv("WINWIN_PUB_PORT", "5570")), "topic": b"winner|"},
    "Elixir": {"port": int(os.getenv("ELIXIR_PUB_PORT", "5560")), "topic": b"ecount|"},
    "Cards": {"port": int(os.getenv("CARD_PUB_PORT", "5590")), "topic": b"cards|"},
    "Troops": {"port": int(os.getenv("TROOP_PUB_PORT", "5580")), "topic": b"troops|"},
}

def test_service(name, port, topic, host="localhost", timeout=5000):
    """Test a single service by subscribing to its topic"""
    print(f"\n{'='*60}")
    print(f"Testing {name} Service on port {port}")
    print(f"{'='*60}")
    
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{host}:{port}")
    socket.subscribe(topic)
    socket.setsockopt(zmq.RCVTIMEO, timeout)
    
    try:
        print(f"Waiting for message from {name}...")
        msg = socket.recv()
        print(f"✓ Received: {msg[:100]}...")  # Print first 100 chars
        return True
    except zmq.Again:
        print(f"✗ Timeout: No message received from {name} within {timeout/1000}s")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        socket.close()
        context.term()

def main():
    """Test all services"""
    print("="*60)
    print("AIO Services Container Test")
    print("="*60)
    print("\nThis script tests connectivity to all 4 services")
    print("Make sure the AIO services container is running!")
    print("\nReading port configuration from .env.aio")
    
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    print(f"\nTesting services on host: {host}")
    print("\nPort Configuration:")
    for name, config in SERVICES.items():
        print(f"  {name:15} Port {config['port']}")
    
    results = {}
    for name, config in SERVICES.items():
        results[name] = test_service(name, config["port"], config["topic"], host)
        time.sleep(0.5)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{name:15} {status}")
    
    print(f"\nTotal: {passed}/{total} services responded")
    
    if passed == total:
        print("\n✓ All services are running correctly!")
        return 0
    else:
        print(f"\n✗ {total - passed} service(s) failed to respond")
        return 1

if __name__ == "__main__":
    sys.exit(main())
