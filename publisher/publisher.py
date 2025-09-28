import cv2
import zmq
import time
import threading

# ZeroMQ PUB socket for sending frames
FREQ = 1
print(f"Publishing at {FREQ} sec interval")
context = zmq.Context()
pub_socket = context.socket(zmq.PUB)
pub_socket.bind("tcp://*:5550")

# Open webcam at index 9
cap = cv2.VideoCapture(9)
if not cap.isOpened():
    raise RuntimeError("Could not open camera index 9")

# --- Receiver function ---
def receiver():
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect("tcp://localhost:5551")  # connect to elixir count publisher
    sub_socket.setsockopt(zmq.SUBSCRIBE, b"ecount|")  # subscribe to elixir count topic

    print("Receiver started. Listening for elixir counts on port 5551")
    while True:
        msg = sub_socket.recv()
        topic, payload = msg.split(b"|", 1)
        count = int(payload.decode())
        print(f"[Receiver] Elixir count received: {count}")





# Run receiver in a separate thread
threading.Thread(target=receiver, daemon=True).start()




# --- Main publishing loop ---
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
