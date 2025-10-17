import os
import zmq
import json
from dotenv import load_dotenv
from pathlib import Path
import time


# Load environment variables from masterreceiver.env
script_dir = Path(__file__).parent.absolute()
env_path = script_dir / ".env"
load_dotenv(env_path)

# Configuration from environment
TROOPS_PORT = os.getenv("TROOPS_PORT", "5580")
ELIXIR_PORT = os.getenv("ELIXIR_PORT", "5560")
WIN_PORT = os.getenv("WIN_PORT", "5570")
CARDS_PORT = os.getenv("CARDS_PORT", "5590")  # Added card detection port

# Log file for standardized JSON
log_file_path = script_dir / "game_data_log.jsonl"

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

# Current state for logging
current_state = {
    "elixir": 0,
    "win_detection": "ongoing",  # Can be: "ongoing", True, or False
    "cards_in_hand": [],
    "troops": []
}

# Function to log current state to JSONL
def log_state(state):
    current_state["timestamp"] = int(time.time() * 1000)

    with open(log_file_path, "a") as f:
        json.dump(state, f)
        f.write("\n")

# Processing functions
def process_troops(msg):
    """Process troop detection message"""
    global current_state
    try:
        topic, json_data = msg.split(b"|", 1)
        results = json.loads(json_data.decode())
        
        troops_list = []
        if isinstance(results, list) and results:
            for result in results:
                predictions = result.get("predictions", {}).get("predictions", [])
                for pred in predictions:
                    class_name = pred.get("class", "Unknown")
                    team = "enemy" if "enemy" in class_name.lower() else "ally" if "ally" in class_name.lower() else "unknown"
                    troops_list.append({
                        "type": class_name,
                        "confidence": float(pred.get("confidence", 0)),
                        "x": float(pred.get("x", 0)),
                        "y": float(pred.get("y", 0)),
                        "team": team
                    })
        current_state["troops"] = troops_list

        # Detailed troop listing
        print(f"Troops: {len(troops_list)}:")
        for i, troop in enumerate(troops_list, 1):
            ttype = troop.get('type', 'Unknown')
            conf = troop.get('confidence', 0.0)
            x = troop.get('x', 0.0)
            y = troop.get('y', 0.0)
            team = troop.get('team', 'unknown')
            print(f"  {i}. {ttype} ({conf:.3f}) at ({x:.1f}, {y:.1f}) team: {team}")
        print()  # blank line for readability
        log_state(current_state)
    except json.JSONDecodeError as e:
        print(f"❌ JSON Error: {e}")

def process_elixir(msg):
    """Process elixir count message"""
    global current_state
    topic, payload = msg.split(b"|", 1)
    current_state["elixir"] = int(payload.decode())
    # Compact output
    print(f"elixir:{current_state['elixir']}")
    log_state(current_state)

def process_win(msg):
    """Process win/lose/ongoing message"""
    global current_state
    topic, result_data = msg.split(b"|", 1)
    result_str = result_data.decode()
    
    # Handle three states: "True", "False", or "ongoing"
    if result_str == "True":
        current_state["win_detection"] = True
    elif result_str == "False":
        current_state["win_detection"] = False
    elif result_str == "ongoing":
        current_state["win_detection"] = "ongoing"
    else:
        # Unexpected value, keep as ongoing
        current_state["win_detection"] = "ongoing"

    # Compact output
    print(f"wincondition:{current_state['win_detection']}")
    log_state(current_state)

def process_cards(msg):
    """Process card detection message"""
    global current_state
    topic, cards_data = msg.split(b"|", 1)
    cards_str = cards_data.decode()
    cards_list = []

    if cards_str:
        for card_info in cards_str.split(","):
            if ":" in card_info:
                slot, name = card_info.split(":", 1)
                cards_list.append({"slot": int(slot), "name": name})
    current_state["cards_in_hand"] = cards_list

    # Detailed cards listing
    print()
    print("Cards:")
    if not cards_list:
        print("  No cards detected")
    else:
        for i, c in enumerate(cards_list, 1):
            name = c.get('name', 'Unknown')
            print(f"  {i}. {name}")
    log_state(current_state)

# Main loop
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

        if sub_socket_cards in socks:
            msg = sub_socket_cards.recv()
            process_cards(msg)

except KeyboardInterrupt:
    print("\nShutting down receiver...")
finally:
    sub_socket_troops.close()
    sub_socket_elixir.close()
    sub_socket_win.close()
    sub_socket_cards.close()
    context.term()
