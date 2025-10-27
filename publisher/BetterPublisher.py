import cv2
import zmq
import time
import argparse
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description="Frame Publisher")
parser.add_argument("--image", action="store_true", help="Use ref.png instead of camera")
args = parser.parse_args()

FREQ = 1
print(f"Publishing at {FREQ} sec interval")

# ZeroMQ PUB socket for sending frames
context = zmq.Context()
pub_socket = context.socket(zmq.PUB)
pub_socket.bind("tcp://*:5550")

# Check if using image mode
if args.image:
    image_path = Path(__file__).parent / "ref.png"
    if not image_path.exists():
        print(f"Error: {image_path} not found!")
        exit(1)
    
    print(f"Using image: {image_path}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        print("Error: Failed to load image!")
        exit(1)
    
    # Main publishing loop for image
    print("Publishing image continuously...")
    while True:
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            print("Failed to encode image")
            continue

        pub_socket.send(b"frame|" + buffer.tobytes())
        time.sleep(FREQ)

# Camera mode (original code)
else:
    # Try camera index 9 first
    camera_index = 9
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Camera index {camera_index} not available.")
        
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            cams = graph.get_input_devices()  # list of camera names
            available = []
            print("Available cameras:")
            for idx, name in enumerate(cams):
                test_cap = cv2.VideoCapture(idx)
                if test_cap.isOpened():
                    available.append(idx)
                    print(f"{idx}: {name}")
                    test_cap.release()
            if not available:
                raise RuntimeError("No cameras detected.")
        except ImportError:
            print("pygrabber not installed, listing indices only.")
            available = [i for i in range(10) if cv2.VideoCapture(i).isOpened()]
            for idx in available:
                print(f"Index {idx}")

        while True:
            try:
                camera_index = int(input("Enter camera index to use: "))
                if camera_index in available:
                    cap = cv2.VideoCapture(camera_index)
                    if cap.isOpened():
                        break
                    else:
                        print("Failed to open selected camera. Try again.")
                else:
                    print("Invalid index. Choose from the available cameras.")
            except ValueError:
                print("Please enter a valid integer.")

    print(f"Using camera index {camera_index}")

    # Main publishing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue

        pub_socket.send(b"frame|" + buffer.tobytes())
        time.sleep(FREQ)
