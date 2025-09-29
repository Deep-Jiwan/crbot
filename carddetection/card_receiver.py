import zmq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PUB_PORT = os.getenv("PUB_PORT", "5554")
ZMQ_ADDRESS = f"tcp://localhost:{PUB_PORT}"

def main():
    """Receive and print card detection results."""
    print(f"Connecting to card publisher at {ZMQ_ADDRESS}")
    
    # ZeroMQ context and subscriber
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(ZMQ_ADDRESS)
    sub_socket.setsockopt(zmq.SUBSCRIBE, b"cards|")
    
    print("Listening for card detection results...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            # Receive message
            msg = sub_socket.recv()
            topic, cards_data = msg.split(b"|", 1)
            
            # Parse card data
            cards_str = cards_data.decode()
            if cards_str:
                # Parse format: "1:CardName1,2:CardName2,3:CardName3,4:CardName4"
                cards = []
                for card_info in cards_str.split(","):
                    if ":" in card_info:
                        idx, name = card_info.split(":", 1)
                        cards.append((int(idx), name))
                
                print(f"🎴 Detected Cards: {cards}")
            else:
                print("🎴 No cards detected")
                
    except KeyboardInterrupt:
        print("\nShutting down receiver...")
    finally:
        sub_socket.close()
        context.term()

if __name__ == "__main__":
    main()
