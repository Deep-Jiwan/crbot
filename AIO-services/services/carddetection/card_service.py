import os
import cv2
import threading
import time
import zmq
import numpy as np
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
FRAME_WIDTH = 1080
FRAME_HEIGHT = 1920

IMAGE_FOLDER = "services/carddetection/images"
CARDS_FOLDER = "services/carddetection/cards"

# Card slots (x1, y1, x2, y2)
CARD_SLOTS = {
    "slot1": (225, 1560, 415, 1785),
    "slot2": (415, 1560, 605, 1785),
    "slot3": (605, 1560, 795, 1785),
    "slot4": (795, 1560, 985, 1785),
}

# Inference API configuration - Card Detection specific
ROBOFLOW_URL = os.getenv("CARD_DETECTION_ROBOFLOW_URL") or os.getenv("ROBOFLOW_URL")
ROBOFLOW_API_KEY = os.getenv("CARD_DETECTION_ROBOFLOW_API_KEY") or os.getenv("ROBOFLOW_API_KEY")
WORKSPACE_NAME = os.getenv("CARD_DETECTION_WORKSPACE_NAME") or os.getenv("WORKSPACE_NAME")
WORKFLOW_ID = os.getenv("CARD_DETECTION_WORKFLOW_ID") or os.getenv("WORKFLOW_ID")

# ZeroMQ configuration
ZMQ_ADDRESS = os.getenv("ZMQ_ADDRESS", "tcp://localhost:5550")
PUB_PORT = os.getenv("CARD_PUB_PORT") or os.getenv("PUB_PORT", "5554")

# Validate required environment variables
if not ROBOFLOW_API_KEY:
    raise ValueError("CARD_DETECTION_ROBOFLOW_API_KEY (or ROBOFLOW_API_KEY) environment variable is required")
if not WORKSPACE_NAME:
    raise ValueError("CARD_DETECTION_WORKSPACE_NAME (or WORKSPACE_NAME) environment variable is required")
if not WORKFLOW_ID:
    raise ValueError("CARD_DETECTION_WORKFLOW_ID (or WORKFLOW_ID) environment variable is required")

# Shared state for latest detected cards
latest_cards = []
cards_lock = threading.Lock()

# Ensure folders exist
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CARDS_FOLDER, exist_ok=True)

# ----------------------------
# ZeroMQ Frame Receiver
# ----------------------------
class FrameReceiver:
    def __init__(self):
        self.context = zmq.Context()
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(ZMQ_ADDRESS)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"frame|")
        # Set socket to non-blocking to get latest frame
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
    
    def get_latest_frame(self):
        """Gets the latest frame from ZeroMQ and saves it."""
        try:
            # Get all available messages to ensure we have the latest
            latest_msg = None
            while True:
                try:
                    msg = self.sub_socket.recv(zmq.NOBLOCK)
                    latest_msg = msg
                except zmq.Again:
                    break
            
            if latest_msg is None:
                return False
                
            topic, jpg_bytes = latest_msg.split(b"|", 1)
            
            img = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print("[CARDS] Failed to decode frame")
                return False
            
            img_resized = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
            
            save_path = os.path.join(IMAGE_FOLDER, "latest_frame.jpg")
            cv2.imwrite(save_path, img_resized)
            
            return True
            
        except Exception as e:
            print(f"[CARDS] Error receiving frame: {e}")
            return False
    
    def close(self):
        """Clean up ZeroMQ resources."""
        self.sub_socket.close()
        self.context.term()

# -----------------------------
# Card Extraction
# -----------------------------
def extract_cards_from_frame(frame_path: str):
    """Crops cards from a frame and saves them in CARDS_FOLDER."""
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Could not read frame: {frame_path}")

    h, w, _ = frame.shape
    if h != FRAME_HEIGHT or w != FRAME_WIDTH:
        print(f"[CARDS] ⚠️ Warning: Frame is {w}x{h}, expected {FRAME_WIDTH}x{FRAME_HEIGHT}")

    for slot_name, (x1, y1, x2, y2) in CARD_SLOTS.items():
        crop = frame[y1:y2, x1:x2]
        save_path = os.path.join(CARDS_FOLDER, f"{slot_name}.png")
        cv2.imwrite(save_path, crop)

# -----------------------------
# Card Detection
# -----------------------------
class CardDetector:
    def __init__(self, cards_folder, roboflow_url, roboflow_api_key, workspace_name, workflow_id):
        self.client = InferenceHTTPClient(
            api_url=roboflow_url,
            api_key=roboflow_api_key,
        )
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        self.cards_folder = cards_folder
        self.latest_results = []

    def detect_cards(self, latest_cards, cards_lock):
        """Detects all card images in the folder."""
        card_files = sorted([
            f for f in os.listdir(self.cards_folder)
            if os.path.isfile(os.path.join(self.cards_folder, f))
               and f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        detected_cards = []
        for idx, filename in enumerate(card_files, start=1):
            path = os.path.join(self.cards_folder, filename)
            try:
                results = self.client.run_workflow(
                    workspace_name=self.workspace_name,
                    workflow_id=self.workflow_id,
                    images={"image": path},
                )

                predictions = []
                if isinstance(results, list) and results:
                    preds_dict = results[0].get("predictions", {})
                    if isinstance(preds_dict, dict):
                        predictions = preds_dict.get("predictions", [])

                if predictions:
                    card_name = predictions[0].get("class", "Unknown")
                else:
                    card_name = "Unknown"

                detected_cards.append((idx, card_name))
            except Exception as e:
                print(f"[CARDS] Error detecting card at index {idx}: {e}")
                detected_cards.append((idx, "Unknown"))

        # Update global shared state
        with cards_lock:
            latest_cards.clear()
            latest_cards.extend(detected_cards)

def publish_cards(pub_socket, cards):
    """Publish card detection results."""
    # Convert cards list to string format
    cards_str = ",".join([f"{idx}:{name}" for idx, name in cards])
    message = f"cards|{cards_str}".encode()
    pub_socket.send(message)
    print(f"[CARDS] Published: {message}")

# -----------------------------
# Card Extraction Thread
# -----------------------------
def card_extraction_loop(frame_receiver, pub_socket):
    """Continuously receive frames and detect cards."""
    detector = CardDetector(CARDS_FOLDER, ROBOFLOW_URL, ROBOFLOW_API_KEY, WORKSPACE_NAME, WORKFLOW_ID)
    frame_path = os.path.join(IMAGE_FOLDER, "latest_frame.jpg")
    
    while True:
        try:
            # Get latest frame from ZeroMQ
            if frame_receiver.get_latest_frame():
                # Extract cards from the updated frame
                extract_cards_from_frame(frame_path)
                detector.detect_cards(latest_cards, cards_lock)
                with cards_lock:
                    print(f"[CARDS] Latest detected cards: {latest_cards}")
                    # Publish card results
                    publish_cards(pub_socket, latest_cards)
        except Exception as e:
            print(f"[CARDS] Error extracting/detecting cards: {e}")

        time.sleep(0.1)

def run_card_service(context):
    """Main service for card detection"""
    # Setup publisher
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://*:{PUB_PORT}")
    
    # Setup frame receiver
    frame_receiver = FrameReceiver()
    
    print(f"[CARDS] Service started")
    print(f"[CARDS] Receiving frames from {ZMQ_ADDRESS}")
    print(f"[CARDS] Publishing on port {PUB_PORT}")
    print(f"[CARDS] Roboflow: {ROBOFLOW_URL}, Workspace: {WORKSPACE_NAME}, Workflow: {WORKFLOW_ID}")
    
    # Run card extraction loop
    card_extraction_loop(frame_receiver, pub_socket)
