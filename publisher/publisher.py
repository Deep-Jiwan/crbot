import cv2
import zmq
import time

FREQ = 1
print(f"Publishing at {FREQ} sec interval")

# ZeroMQ PUB socket for sending frames
context = zmq.Context()
pub_socket = context.socket(zmq.PUB)
pub_socket.bind("tcp://*:5550")

# Open webcam at index 9
cap = cv2.VideoCapture(9)
if not cap.isOpened():
    raise RuntimeError("Could not open camera index 9")

# Main publishing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    # Encode frame as JPEG
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        continue

    # Send frame with topic
    pub_socket.send(b"frame|" + buffer.tobytes())
    time.sleep(FREQ)
