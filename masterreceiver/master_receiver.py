import os
import zmq
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from masterreceiver.env
script_dir = Path(__file__).parent.absolute()
env_path = script_dir / "masterreceiver.env"
load_dotenv(env_path)

# Configuration from environment
TROOPS_PORT = os.getenv("TROOPS_PORT", "5555")
ELIXIR_PORT = os.getenv("ELIXIR_PORT", "5551")
WIN_PORT = os.getenv("WIN_PORT", "5570")
CARDS_PORT = os.getenv("CARDS_PORT", "5554")  # Added card detection port

# Create ZeroMQ context
context = zmq.Context()

# Create sockets
sub_socket_troops = context.socket(zmq.SUB)
sub_socket_troops.connect(f"tcp://localhost:{TROOPS_PORT}")
sub_socket_troops.setsockopt(zmq.SUBSCRIBE, b"troops|")

sub_socket_elixir = context.socket(zmq.SUB)
sub_socket_elixir.connect(f"tcp://localhost:{ELIXIR_PORT}")
sub_socket_elixir.setsockopt(zmq.SUBSCRIBE, b"ecount|")

sub_socket_win = context.socket(zmq.SUB)
sub_socket_win.connect(f"tcp://localhost:{WIN_PORT}")
sub_socket_win.setsockopt(zmq.SUBSCRIBE, b"winner|")

sub_socket_cards = context.socket(zmq.SUB)  # Card detection socket
sub_socket_cards.connect(f"tcp://localhost:{CARDS_PORT}")
sub_socket_cards.setsockopt(zmq.SUBSCRIBE, b"cards|")

# Set up poller
poller = zmq.Poller()
poller.register(sub_socket_troops, zmq.POLLIN)
poller.register(sub_socket_elixir, zmq.POLLIN)
poller.register(sub_socket_win, zmq.POLLIN)
poller.register(sub_socket_cards, zmq.POLLIN)  # Added card socket

# Processing functions
def process_troops(msg):
    """Process troop detection message"""
    try:
        topic, json_data = msg.split(b"|", 1)
        results = json.loads(json_data.decode())
        
        print("\n" + "="*50)
        print("🏰 TROOP DETECTION RESULTS")
        print("="*50)
        
        if isinstance(results, list) and results:
            result = results[0]
            if "count_objects" in result:
                print(f"📊 Total Objects: {result['count_objects']}")
            if "predictions" in result and "predictions" in result["predictions"]:
                predictions = result["predictions"]["predictions"]
                print(f"🎯 Predictions: {len(predictions)}")
                for i, pred in enumerate(predictions, 1):
                    class_name = pred.get("class", "Unknown")
                    confidence = pred.get("confidence", 0)
                    x, y = pred.get("x", 0), pred.get("y", 0)
                    print(f"  {i}. {class_name} ({confidence:.3f}) at ({x}, {y})")
        else:
            print("📋 Raw Results:", json.dumps(results, indent=2))
    except json.JSONDecodeError as e:
        print(f"❌ JSON Error: {e}")

def process_elixir(msg):
    """Process elixir count message"""
    topic, payload = msg.split(b"|", 1)
    count = int(payload.decode())
    print(f"\n🧪 ELIXIR COUNT: {count}")

def process_win(msg):
    """Process win/lose message"""
    topic, result_data = msg.split(b"|", 1)
    result_str = result_data.decode()
    is_win = result_str == "True"
    print(f"\n{'🏆 WIN' if is_win else '💀 LOSE'} - {'Victory' if is_win else 'Defeat'} detected!")

def process_cards(msg):
    """Process card detection message"""
    topic, cards_data = msg.split(b"|", 1)
    cards_str = cards_data.decode()
    if cards_str:
        # Format: "1:CardName1,2:CardName2,3:CardName3,4:CardName4"
        cards = []
        for card_info in cards_str.split(","):
            if ":" in card_info:
                idx, name = card_info.split(":", 1)
                cards.append((int(idx), name))
        print(f"\n🎴 DETECTED CARDS: {cards}")
    else:
        print("\n🎴 No cards detected")

print("Master Receiver started. Listening for messages...")
print("Press Ctrl+C to exit")

try:
    while True:
        socks = dict(poller.poll())
        
        if sub_socket_troops in socks:
            msg = sub_socket_troops.recv()
            process_troops(msg)
            
        if sub_socket_elixir in socks:
            msg = sub_socket_elixir.recv()
            process_elixir(msg)
            
        if sub_socket_win in socks:
            msg = sub_socket_win.recv()
            process_win(msg)
            
        if sub_socket_cards in socks:  # Card detection handling
            msg = sub_socket_cards.recv()
            process_cards(msg)
            
except KeyboardInterrupt:
    print("\nShutting down receiver...")
finally:
    sub_socket_troops.close()
    sub_socket_elixir.close()
    sub_socket_win.close()
    sub_socket_cards.close()  # Close card socket
    context.term()
