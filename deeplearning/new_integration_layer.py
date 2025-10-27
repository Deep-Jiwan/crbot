"""
Simple Integration Layer for Clash Royale AI
Checks if services are running, then starts.
"""

import sys
import os
import argparse
import random
import time
import zmq
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add gameplayer to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gameplayer"))
from gameplayer import GamePlayer


# Services to check from .env
SERVICES = [
    ("Publisher", int(os.getenv("STATE_PUBLISHER_PORT", "5550")), True, "zmq"),
    ("Elixir", int(os.getenv("ELIXIR_PORT", "5560")), True, "zmq"),
    ("Cards", int(os.getenv("CARDS_PORT", "5590")), True, "zmq"),
    ("Troops", int(os.getenv("TROOPS_PORT", "5580")), True, "zmq"),
    ("Win", int(os.getenv("WIN_PORT", "5570")), True, "zmq"),
    ("Inference", os.getenv("INFERENCE_LINK", "http://localhost:9001"), True, "http"),
]


def check_zmq_service(port, timeout=1000):
    """Check if a ZMQ service is running on the port"""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVTIMEO, timeout)
    
    try:
        socket.connect(f"tcp://localhost:{port}")
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.recv()
        return True
    except:
        return False
    finally:
        socket.close()
        context.term()


def check_http_service(url, timeout=2):
    """Check if an HTTP service is running"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False


def check_all_services():
    """Check all services and return if all required ones are running"""
    print("\nChecking services...")
    all_ok = True
    
    for name, port_or_url, required, service_type in SERVICES:
        if service_type == "zmq":
            status = check_zmq_service(port_or_url)
            location = f"port {port_or_url}"
        else:  # http
            status = check_http_service(port_or_url)
            location = port_or_url
        
        mark = "✓" if status else "✗"
        req = "(required)" if required else "(optional)"
        print(f"  {mark} {name} on {location} {req}")
        
        if required and not status:
            all_ok = False
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Clash Royale AI Integration")
    parser.add_argument("--no-checks", action="store_true", 
                       help="Skip service checks")
    parser.add_argument("--test", action="store_true",
                       help="Run GamePlayer test")
    args = parser.parse_args()
    
    # Check services unless --no-checks
    if not args.no_checks:
        if not check_all_services():
            print("\n✗ Required services not running. Start them first.")
            print("  Or use --no-checks to skip this check.\n")
            sys.exit(1)
        print("\n✓ All required services running!\n")
    else:
        print("\n⚠ Skipping service checks (--no-checks)\n")
    
    # Initialize GamePlayer
    print("Initializing GamePlayer...")
    player = GamePlayer()
    
    # Run test if requested
    if args.test:
        player.test_functionality()
    else:
        # TODO: Add your AI code here
        print("Ready to start AI...")
        print("Use --test flag to test GamePlayer functionality")


if __name__ == "__main__":
    main()


# usage of game player 
# # Deploy card 2 at center arena
# player.place_card(2, 1600, 500) (card,x,y)
