import pyautogui
import time
import threading
from pynput import mouse, keyboard
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CoordinateCapture:
    """
    A utility class for capturing screen coordinates by clicking.
    More robust than the basic implementation in GamePlayer.
    """

    def __init__(self):
        self.capture_mode = False
        self.mouse_listener = None
        self.keyboard_listener = None
        self.coordinates_file = os.path.join(os.path.dirname(__file__), "captured_coordinates.txt")

    def start_capture(self):
        """Start coordinate capture mode"""
        if self.capture_mode:
            print("Coordinate capture mode is already active. Press 'q' to quit.")
            return

        print("\n" + "="*50)
        print("COORDINATE CAPTURE MODE ACTIVE")
        print("Click anywhere to capture coordinates")
        print("Press 'q' to quit capture mode")
        print("="*50 + "\n")

        self.capture_mode = True

        # Start mouse and keyboard listeners
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)

        self.mouse_listener.start()
        self.keyboard_listener.start()

        print("Click anywhere to capture coordinates...")

    def on_click(self, x, y, button, pressed):
        """Handle mouse click events"""
        if self.capture_mode and pressed:
            coords = (int(x), int(y))
            print(f"Captured coordinates: {coords}")

            # Save to file
            try:
                with open(self.coordinates_file, "a") as f:
                    f.write(f"{coords}\n")
            except Exception as e:
                print(f"Error saving coordinates to file: {e}")

    def on_press(self, key):
        """Handle keyboard press events"""
        try:
            if key.char == 'q' and self.capture_mode:
                self.stop_capture()
        except AttributeError:
            pass  # Handle special keys

    def stop_capture(self):
        """Stop coordinate capture mode"""
        if not self.capture_mode:
            return

        self.capture_mode = False

        # Stop listeners
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()

        print("\nCoordinate capture mode ended.")
        print(f"Captured coordinates saved to: {self.coordinates_file}")

        # Show recent captures
        try:
            with open(self.coordinates_file, "r") as f:
                lines = f.readlines()
                recent = lines[-5:] if len(lines) > 5 else lines
                if recent:
                    print("\nRecent captures:")
                    for line in recent:
                        print(f"  {line.strip()}")
        except Exception as e:
            print(f"Error reading captured coordinates: {e}")


# Standalone coordinate capture tool
if __name__ == "__main__":
    capturer = CoordinateCapture()
    try:
        capturer.start_capture()

        # Keep the program running
        while capturer.capture_mode:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nProgram interrupted")
    finally:
        capturer.stop_capture()
