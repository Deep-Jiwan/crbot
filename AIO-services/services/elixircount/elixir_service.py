import os
import cv2
import numpy as np
import zmq
import time
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- Configuration from ENV ---
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 1080))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 1920))
ELIXIR_Y = int(os.getenv("ELIXIR_Y", 1810))
ELIXIR_X_START = int(os.getenv("ELIXIR_X_START", 340))
ELIXIR_X_END = int(os.getenv("ELIXIR_X_END", 990))
ELIXIR_STEP = int(os.getenv("ELIXIR_STEP", 70))
TARGET_B = int(os.getenv("TARGET_B", 241))
TARGET_G = int(os.getenv("TARGET_G", 119))
TARGET_R = int(os.getenv("TARGET_R", 233))
TOLERANCE = int(os.getenv("TOLERANCE", 80))
ZMQ_ADDRESS = os.getenv("ZMQ_ADDRESS", "tcp://localhost:5550")
PUB_PORT = os.getenv("ELIXIR_PUB_PORT") or os.getenv("PUB_PORT", "5551")
SLEEP_TIME = float(os.getenv("SLEEP_TIME", 0.1))
ANNOTATE = os.getenv("ANNOTATE", "True")

IMAGE_FOLDER = os.getenv("IMAGE_FOLDER", "services/elixircount/images")
ANNOTATED_FOLDER = os.getenv("ANNOTATED_FOLDER", "services/elixircount/frame")

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)
frame_filename = os.path.join(IMAGE_FOLDER, "latest_frame.jpg")

# --- Functions ---
def process_frame(img):
    count = 0
    elixir_positions = []

    for x in range(ELIXIR_X_START, ELIXIR_X_END, ELIXIR_STEP):
        b, g, r = [int(v) for v in img[ELIXIR_Y, x]]
        if (abs(b - TARGET_B) <= TOLERANCE and
            abs(g - TARGET_G) <= TOLERANCE and
            abs(r - TARGET_R) <= TOLERANCE):
            count += 1
            elixir_positions.append((x, ELIXIR_Y))

    # Annotate image
    if(ANNOTATE == "True"):
        print(f"[ELIXIR] Elixir Count: {count}")
        annotated_frame = img.copy()
        for x, y in elixir_positions:
            cv2.rectangle(annotated_frame, (x-10, y-10), (x+10, y+10), (0, 0, 255), 3)

        annotated_path = os.path.join(ANNOTATED_FOLDER, "latest_annotated.jpg")
        cv2.imwrite(annotated_path, annotated_frame)

    return count

def publish_count(pub_socket, count):
    message = f"ecount|{count}".encode()
    pub_socket.send(message)
    print(f"[ELIXIR] Published: {message}")

def run_elixir_service(context):
    """Main service loop for elixir counting"""
    # Subscriber for frames
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(ZMQ_ADDRESS)
    sub_socket.setsockopt(zmq.SUBSCRIBE, b"frame|")

    # Publisher for elixir count
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://*:{PUB_PORT}")

    print(f"[ELIXIR] Service started")
    print(f"[ELIXIR] Receiving frames from {ZMQ_ADDRESS}")
    print(f"[ELIXIR] Publishing on port {PUB_PORT}")
    print(f"[ELIXIR] Annotation set to: {ANNOTATE}")

    # Main loop
    while True:
        msg = sub_socket.recv()
        topic, jpg_bytes = msg.split(b"|", 1)

        img = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("[ELIXIR] Failed to decode frame")
            continue

        img_resized = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        cv2.imwrite(frame_filename, img_resized)

        ecount = process_frame(img_resized)
        publish_count(pub_socket, ecount)

        time.sleep(SLEEP_TIME)
