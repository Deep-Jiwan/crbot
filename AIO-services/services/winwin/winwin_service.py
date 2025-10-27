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

# Winner detection parameters
WINNER_Y1 = int(os.getenv("WINNER_Y1", 267))
WINNER_Y2 = int(os.getenv("WINNER_Y2", 800))
WINNER_X1 = int(os.getenv("WINNER_X1", 400))
WINNER_X2 = int(os.getenv("WINNER_X2", 424))

# Target color 1 (BGR) for p1, p2
TARGET_B1 = int(os.getenv("TARGET_B1", 255))
TARGET_G1 = int(os.getenv("TARGET_G1", 204))
TARGET_R1 = int(os.getenv("TARGET_R1", 255))
TOLERANCE1 = int(os.getenv("TOLERANCE1", 50))

# Target color 2 (BGR) for p3, p4
TARGET_B2 = int(os.getenv("TARGET_B2", 255))
TARGET_G2 = int(os.getenv("TARGET_G2", 255))
TARGET_R2 = int(os.getenv("TARGET_R2", 102))
TOLERANCE2 = int(os.getenv("TOLERANCE2", 50))

ZMQ_ADDRESS = os.getenv("ZMQ_ADDRESS", "tcp://172.17.72.88:5550")
PUB_PORT = os.getenv("WINWIN_PUB_PORT") or os.getenv("PUB_PORT", "5570")
SLEEP_TIME = float(os.getenv("SLEEP_TIME", 0.1))
ANNOTATE = os.getenv("ANNOTATE", "True").lower() == "true"

IMAGE_FOLDER = os.getenv("IMAGE_FOLDER", "services/winwin/images")
ANNOTATED_FOLDER = os.getenv("ANNOTATED_FOLDER", "services/winwin/frame")

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)
frame_filename = os.path.join(IMAGE_FOLDER, "latest_frame.jpg")

# Global variable for backoff timing
last_backoff_time = 0

# --- Functions ---
def process_frame(img):
    global last_backoff_time
    
    # Check if we're in backoff period
    current_time = time.time()
    if current_time - last_backoff_time < 1.0:
        return None  # Skip processing during backoff
    
    # Sample pixels at the four specific locations
    # p1(400,267), p2(424,267), p3(400,800), p4(424,800)
    b1, g1, r1 = [int(v) for v in img[WINNER_Y1, WINNER_X1]]  # p1
    b2, g2, r2 = [int(v) for v in img[WINNER_Y1, WINNER_X2]]  # p2
    b3, g3, r3 = [int(v) for v in img[WINNER_Y2, WINNER_X1]]  # p3
    b4, g4, r4 = [int(v) for v in img[WINNER_Y2, WINNER_X2]]  # p4
    
    # Check if p1 and p2 match target1
    p1_match = (abs(b1 - TARGET_B1) <= TOLERANCE1 and
               abs(g1 - TARGET_G1) <= TOLERANCE1 and
               abs(r1 - TARGET_R1) <= TOLERANCE1)
    
    p2_match = (abs(b2 - TARGET_B1) <= TOLERANCE1 and
               abs(g2 - TARGET_G1) <= TOLERANCE1 and
               abs(r2 - TARGET_R1) <= TOLERANCE1)
    
    # Check if p3 and p4 match target2
    p3_match = (abs(b3 - TARGET_B2) <= TOLERANCE2 and
               abs(g3 - TARGET_G2) <= TOLERANCE2 and
               abs(r3 - TARGET_R2) <= TOLERANCE2)
    
    p4_match = (abs(b4 - TARGET_B2) <= TOLERANCE2 and
               abs(g4 - TARGET_G2) <= TOLERANCE2 and
               abs(r4 - TARGET_R2) <= TOLERANCE2)
    
    # Determine result based on logic
    p12_both_match = p1_match and p2_match
    p34_both_match = p3_match and p4_match
    
    result = None
    if p12_both_match and not p34_both_match:
        result = False  # LOSE
    elif not p12_both_match and p34_both_match:
        result = True   # WIN
    else:
        # No clear result - enter backoff
        last_backoff_time = current_time
        if ANNOTATE:
            print("[WINWIN] No clear result - entering 3 second backoff")
            print(f"[WINWIN] p1&p2 match target1: {p12_both_match}, p3&p4 match target2: {p34_both_match}")
        return None
    
    # Annotate image
    if ANNOTATE:
        result_text = "WIN" if result else "LOSE"
        print(f"[WINWIN] Result: {result_text}")
        print(f"[WINWIN] p1({WINNER_X1},{WINNER_Y1}): BGR({b1},{g1},{r1}) - Match T1: {p1_match}")
        print(f"[WINWIN] p2({WINNER_X2},{WINNER_Y1}): BGR({b2},{g2},{r2}) - Match T1: {p2_match}")
        print(f"[WINWIN] p3({WINNER_X1},{WINNER_Y2}): BGR({b3},{g3},{r3}) - Match T2: {p3_match}")
        print(f"[WINWIN] p4({WINNER_X2},{WINNER_Y2}): BGR({b4},{g4},{r4}) - Match T2: {p4_match}")
        
        annotated_frame = img.copy()
        # Mark the sampled pixels with different colors
        color1 = (0, 255, 0) if p1_match else (0, 0, 255)  # Green if match T1, red if not
        color2 = (0, 255, 0) if p2_match else (0, 0, 255)
        color3 = (255, 0, 0) if p3_match else (0, 0, 255)  # Blue if match T2, red if not
        color4 = (255, 0, 0) if p4_match else (0, 0, 255)
        
        cv2.circle(annotated_frame, (WINNER_X1, WINNER_Y1), 5, color1, -1)  # p1
        cv2.circle(annotated_frame, (WINNER_X2, WINNER_Y1), 5, color2, -1)  # p2
        cv2.circle(annotated_frame, (WINNER_X1, WINNER_Y2), 5, color3, -1)  # p3
        cv2.circle(annotated_frame, (WINNER_X2, WINNER_Y2), 5, color4, -1)  # p4
        
        # Add result text
        cv2.putText(annotated_frame, result_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 
                   (0, 255, 0) if result else (0, 0, 255), 3)
        
        annotated_path = os.path.join(ANNOTATED_FOLDER, "latest_annotated.jpg")
        cv2.imwrite(annotated_path, annotated_frame)
    
    return result

def publish_result(pub_socket, result):
    """
    Publish win/lose/ongoing result.
    - result = True: WIN
    - result = False: LOSE
    - result = None: ONGOING
    """
    if result is True:
        result_str = "True"
    elif result is False:
        result_str = "False"
    else:  # None or anything else
        result_str = "ongoing"
    
    message = f"winner|{result_str}".encode()
    pub_socket.send(message)
    print(f"[WINWIN] Published: {message}")

def run_winwin_service(context):
    """Main service loop for winwin detection"""
    # Subscriber for frames
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(ZMQ_ADDRESS)
    sub_socket.setsockopt(zmq.SUBSCRIBE, b"frame|")

    # Publisher for win/lose results
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://*:{PUB_PORT}")

    print(f"[WINWIN] Service started")
    print(f"[WINWIN] Receiving frames from {ZMQ_ADDRESS}")
    print(f"[WINWIN] Publishing on port {PUB_PORT}")
    print(f"[WINWIN] Annotation set to: {ANNOTATE}")

    # Main loop
    while True:
        msg = sub_socket.recv()
        topic, jpg_bytes = msg.split(b"|", 1)

        img = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("[WINWIN] Failed to decode frame")
            continue

        img_resized = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        cv2.imwrite(frame_filename, img_resized)

        result = process_frame(img_resized)
        publish_result(pub_socket, result)

        time.sleep(SLEEP_TIME)
