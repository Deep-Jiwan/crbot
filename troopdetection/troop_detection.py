import os
import cv2
import zmq
import time
import json
import base64
import requests
from dotenv import load_dotenv

# Load environment variables from .env in the same folder
load_dotenv()

# --- Configuration from ENV ---
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 1080))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 1920))
ZMQ_SUB_ADDRESS = os.getenv("ZMQ_SUB_ADDRESS", "tcp://localhost:5500")
ROB_FLOW_API_KEY = os.getenv("ROB_FLOW_API_KEY")
ROB_FLOW_WORKFLOW_URL = os.getenv("ROB_FLOW_WORKFLOW_URL")
SLEEP_TIME = float(os.getenv("SLEEP_TIME", 0.1))

# --- ZeroMQ Subscriber ---
context = zmq.Context()
sub_socket = context.socket(zmq.SUB)
sub_socket.connect(ZMQ_SUB_ADDRESS)
sub_socket.setsockopt(zmq.SUBSCRIBE, b"frame|")

print(f"Subscribed to frames at {ZMQ_SUB_ADDRESS}")

def send_to_roboflow(image):
    """Send an image to the Roboflow workflow and return JSON response"""
    _, img_encoded = cv2.imencode(".jpg", image)
    img_bytes = img_encoded.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    headers = {"Authorization": f"Bearer {ROB_FLOW_API_KEY}"}
    payload = {"image": img_base64}

    response = requests.post(ROB_FLOW_WORKFLOW_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"⚠️ Roboflow API error {response.status_code}: {response.text}")
        return None

# --- Main Loop ---
while True:
    msg = sub_socket.recv()
    topic, jpg_bytes = msg.split(b"|", 1)

    img = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to decode frame")
        continue

    img_resized = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

    result_json = send_to_roboflow(img_resized)
    if result_json:
        print("Roboflow Result:", json.dumps(result_json, indent=2))

    time.sleep(SLEEP_TIME)