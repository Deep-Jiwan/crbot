import os
import zmq
from dotenv import load_dotenv

load_dotenv()

# Get PUB_PORT from environment variables (default 5551)
PUB_PORT = os.getenv("PUB_PORT", "5551")

context = zmq.Context()
sub_socket = context.socket(zmq.SUB)
sub_socket.connect(f"tcp://localhost:{PUB_PORT}")
sub_socket.setsockopt(zmq.SUBSCRIBE, b"ecount|")  # subscribe to elixir count topic

print(f"[Receiver] Listening for elixir counts on port {PUB_PORT}")

while True:
    msg = sub_socket.recv()
    topic, payload = msg.split(b"|", 1)
    count = int(payload.decode())
    print(f"[Receiver] Elixir count received: {count}")
