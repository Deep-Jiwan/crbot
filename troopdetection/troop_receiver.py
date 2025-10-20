import zmq
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PUB_PORT = os.getenv("PUB_PORT", "5555")
ZMQ_ADDRESS = f"tcp://localhost:{PUB_PORT}"

def main():
    """Receive and print troop detection results."""
    print(f"Connecting to troop detection publisher at {ZMQ_ADDRESS}")
    
    # ZeroMQ context and subscriber
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(ZMQ_ADDRESS)
    sub_socket.setsockopt(zmq.SUBSCRIBE, b"troops|")
    
    print("Listening for troop detection results...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            # Receive message
            msg = sub_socket.recv()
            topic, json_data = msg.split(b"|", 1)
            
            # Parse JSON data
            try:
                results = json.loads(json_data.decode())
                
                # Display results in a formatted way
                print("\n" + "="*50)
                print("🏰 TROOP DETECTION RESULTS")
                print("="*50)
                
                if isinstance(results, list) and results:
                    result = results[0]  # Get first result
                    
                    # Show object count
                    if "count_objects" in result:
                        print(f"📊 Total Objects Detected: {result['count_objects']}")
                    
                    # Show predictions
                    if "predictions" in result and "predictions" in result["predictions"]:
                        predictions = result["predictions"]["predictions"]
                        print(f"🎯 Predictions: {len(predictions)}")
                        
                        for i, pred in enumerate(predictions, 1):
                            class_name = pred.get("class", "Unknown")
                            confidence = pred.get("confidence", 0)
                            x, y = pred.get("x", 0), pred.get("y", 0)
                            
                            print(f"  {i}. {class_name} (confidence: {confidence:.3f}) at ({x}, {y})")
                    
                else:
                    print("📋 Raw Results:", json.dumps(results, indent=2))
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                print(f"Raw data: {json_data}")
                
    except KeyboardInterrupt:
        print("\nShutting down receiver...")
    finally:
        sub_socket.close()
        context.term()

if __name__ == "__main__":
    main()
