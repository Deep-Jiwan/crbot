import os
import cv2
import zmq
import time
import json
import copy
from dotenv import load_dotenv
import numpy as np
from inference_sdk import InferenceHTTPClient
# Load environment variables from .env in the same folder
load_dotenv()

# --- Configuration from ENV ---
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 1080))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 1920))
# ZeroMQ configuration
ZMQ_SUB_ADDRESS = os.getenv("ZMQ_SUB_ADDRESS", "tcp://localhost:5500")
PUB_PORT = os.getenv("PUB_PORT", "5560")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_URL = os.getenv("ROBOFLOW_URL", "http://172.17.72.88:9001")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
WORKFLOW_ID = os.getenv("WORKFLOW_ID")
SLEEP_TIME = float(os.getenv("SLEEP_TIME", 0.1))
# Validate required environment variables
if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY environment variable is required")
if not WORKSPACE_NAME:
    raise ValueError("WORKSPACE_NAME environment variable is required")
if not WORKFLOW_ID:
    raise ValueError("WORKFLOW_ID environment variable is required")

# Create temp folder
TEMP_FOLDER = "temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# --- ZeroMQ Subscriber ---
context = zmq.Context()
sub_socket = context.socket(zmq.SUB)
sub_socket.connect(ZMQ_SUB_ADDRESS)
sub_socket.setsockopt(zmq.SUBSCRIBE, b"frame|")
print(f"Subscribed to frames at {ZMQ_SUB_ADDRESS}")

# ----------------------------
# ZeroMQ Publisher
# ----------------------------
def setup_publisher():
    """Setup ZeroMQ publisher for troop detection results."""
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://*:{PUB_PORT}")
    return pub_socket

def publish_troops(pub_socket, results):
    """Publish troop detection results as JSON."""
    if results:
        # Clean results and convert to JSON string
        clean_result = clean_results(results)
        results_json = json.dumps(clean_result)
        message = f"troops|{results_json}".encode()
        pub_socket.send(message)
        print(f"Published troops data (length: {len(results_json)} chars)")

# Global publisher socket
pub_socket = setup_publisher()

# Initialize Roboflow client
client = InferenceHTTPClient(
    api_url=ROBOFLOW_URL,
    api_key=ROBOFLOW_API_KEY,
)

def detect_troops(image):
    """Send an image to the Roboflow workflow and return detection results"""
    # Save image temporarily for inference
    temp_path = os.path.join(TEMP_FOLDER, "temp_frame.jpg")
    cv2.imwrite(temp_path, image)
    
    try:
        results = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": temp_path},
        )
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return results
        
    except Exception as e:
        print(f"⚠️ Roboflow inference failed: {e}")
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None

def clean_results(results):
    """Remove output_image from results to avoid printing large base64 data"""
    if not results:
        return results
    
    # Create a deep copy to avoid modifying original results
    clean_results = copy.deepcopy(results)
    
    # Handle both list and single object formats
    if isinstance(clean_results, list):
        for item in clean_results:
            if isinstance(item, dict) and "output_image" in item:
                del item["output_image"]
    elif isinstance(clean_results, dict) and "output_image" in clean_results:
        del clean_results["output_image"]
    
    return clean_results

# --- Main Loop ---
while True:
    msg = sub_socket.recv()
    topic, jpg_bytes = msg.split(b"|", 1)

    img = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to decode frame")
        continue

    img_resized = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

    result_json = detect_troops(img_resized)
    if result_json:
        # Clean results to remove large base64 image data
        clean_result = clean_results(result_json)
        print("Troop Detection Result:", json.dumps(clean_result, indent=2))
        # Publish results via ZeroMQ
        publish_troops(pub_socket, result_json)

    time.sleep(SLEEP_TIME)