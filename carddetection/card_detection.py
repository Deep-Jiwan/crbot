import os
import cv2
import zmq
from roboflow import Roboflow
from dotenv import load_dotenv
import time

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 1080))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 1920))
CARD_SLOTS = {
    "slot1": (225, 1560, 415, 1785),
    "slot2": (415, 1560, 605, 1785),
    "slot3": (605, 1560, 795, 1785),
    "slot4": (795, 1560, 985, 1785),
}

IMAGE_FOLDER = os.getenv("IMAGE_FOLDER", "images")
CARDS_FOLDER = os.path.join(os.path.dirname(__file__), "cards")
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CARDS_FOLDER, exist_ok=True)

ZMQ_ADDRESS = os.getenv("ZMQ_ADDRESS", "tcp://localhost:5550")  # frames come in here
PUB_ADDRESS = os.getenv("ZMQ_PUB_ADDRESS", "tcp://*:5552")      # predictions go out here
SLEEP_TIME = float(os.getenv("SLEEP_TIME", 0.1))

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID")

# -------------------------
# ZeroMQ Setup
# -------------------------
context = zmq.Context()

# Subscriber for frames
sub_socket = context.socket(zmq.SUB)
sub_socket.connect(ZMQ_ADDRESS)
sub_socket.setsockopt(zmq.SUBSCRIBE, b"frame|")

# Publisher for card predictions
pub_socket = context.socket(zmq.PUB)
pub_socket.bind(PUB_ADDRESS)

print(f"Card Detection Subscriber listening on {ZMQ_ADDRESS}")
print(f"Publishing predictions on {PUB_ADDRESS}")

# -------------------------
# Roboflow Setup
# -------------------------
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(ROBOFLOW_WORKFLOW_ID)  # use workflow ID
print("Connected to Roboflow workflow:", ROBOFLOW_WORKFLOW_ID)

# -------------------------
# Functions
# -------------------------
def extract_cards(frame):
    """Crop the 4 card slots and save to disk"""
    slots = {}
    for slot_name, (x1, y1, x2, y2) in CARD_SLOTS.items():
        crop = frame[y1:y2, x1:x2]
        save_path = os.path.join(CARDS_FOLDER, f"{slot_name}.png")
        cv2.imwrite(save_path, crop)
        slots[slot_name] = save_path
    return slots

def detect_cards(frame_path):
    """Send cropped cards to Roboflow for prediction"""
    slots = extract_cards(cv2.imread(frame_path))
    results = {}
    for slot_name, img_path in slots.items():
        try:
            prediction = project.predict(img_path)
            results[slot_name] = prediction.json()  # returns dict of prediction data
        except Exception as e:
            results[slot_name] = {"error": str(e)}
    return results

def publish_results(results):
    pub_socket.send_json({"type": "cards", "data": results})
    print("Published card predictions:", results)

# -------------------------
# Main Loop
# -------------------------
frame_file = os.path.join(IMAGE_FOLDER, "latest_frame.png")

while True:
    try:
        msg = sub_socket.recv()
        topic, jpg_bytes = msg.split(b"|", 1)

        img = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to decode frame")
            continue

        # Resize to match expected frame size
        img_resized = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        cv2.imwrite(frame_file, img_resized)

        # Run card detection
        results = detect_cards(frame_file)
        publish_results(results)

        time.sleep(SLEEP_TIME)

    except KeyboardInterrupt:
        print("Exiting card detection...")
        break
