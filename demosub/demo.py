import zmq
import time
import threading
import random
import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
SUB_ADDRESS = os.getenv("SUB_ADDRESS", "tcp://localhost:5551")
PUB_PORT = os.getenv("PUB_PORT", "5552")
TOPIC = os.getenv("TOPIC", "ecount")
SLEEP_TIME = float(os.getenv("SLEEP_TIME", 1))

# --- ZeroMQ context ---
context = zmq.Context()

# Subscriber socket (receives from publisher)
sub_socket = context.socket(zmq.SUB)
sub_socket.connect(SUB_ADDRESS)
sub_socket.setsockopt(zmq.SUBSCRIBE, f"{TOPIC}|".encode())

# Publisher socket (sends its own data)
pub_socket = context.socket(zmq.PUB)
pub_socket.bind(f"tcp://*:{PUB_PORT}")

print(f"Subscriber listening on {SUB_ADDRESS}, publisher sending on port {PUB_PORT}")

# --- Function to receive messages ---
def subscriber_loop():
    while True:
        msg = sub_socket.recv()
        topic, data_bytes = msg.split(b"|", 1)
        data = data_bytes.decode()
        print(f"[Received] Topic: {topic.decode()}, Data: {data}")

# --- Function to publish sample messages ---
def publisher_loop():
    while True:
        sample_count = random.randint(0, 10)
        message = f"{TOPIC}|{sample_count}".encode()
        pub_socket.send(message)
        print(f"[Published] {message}")
        time.sleep(SLEEP_TIME)

# --- Run both loops in separate threads ---
threading.Thread(target=subscriber_loop, daemon=True).start()
threading.Thread(target=publisher_loop, daemon=True).start()

# Keep the main thread alive
while True:
    time.sleep(0.1)
