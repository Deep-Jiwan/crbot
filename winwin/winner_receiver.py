import zmq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PUB_PORT = os.getenv("PUB_PORT", "5570")
ZMQ_ADDRESS = f"tcp://localhost:{PUB_PORT}"

def main():
    """Receive and print win/lose results."""
    print(f"Connecting to winner publisher at {ZMQ_ADDRESS}")
    
    # ZeroMQ context and subscriber
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(ZMQ_ADDRESS)
    sub_socket.setsockopt(zmq.SUBSCRIBE, b"winner|")
    
    print("Listening for win/lose results...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            # Receive message
            msg = sub_socket.recv()
            topic, result_data = msg.split(b"|", 1)
            
            # Parse result
            result_str = result_data.decode()
            is_win = result_str == "True"
            
            # Display result with emoji
            if is_win:
                print("🏆 WIN - Victory detected!")
            else:
                print("💀 LOSE - Defeat detected!")
                
    except KeyboardInterrupt:
        print("\nShutting down receiver...")
    finally:
        sub_socket.close()
        context.term()

if __name__ == "__main__":
    main()
