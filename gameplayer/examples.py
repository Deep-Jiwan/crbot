#!/usr/bin/env python3
"""
Example script demonstrating GamePlayer usage for Clash Royale automation.

This script shows how to integrate the GamePlayer with the existing bot modules
to create a complete automated gameplay system.
"""

import time
from gameplayer import GamePlayer
from coordinate_capture import CoordinateCapture

def example_basic_gameplay():
    """Example of basic gameplay automation"""
    print("=== Basic Gameplay Example ===")

    player = GamePlayer()

    # Start a match
    print("Starting match...")
    player.start_match()
    time.sleep(3)  # Wait for match to load

    # Place some cards
    print("Placing cards...")
    player.place_card(0, 1419, 650)  # Place card 0 at (1419, 650)
    time.sleep(3)
    player.place_card(1, 1419, 650)  # Place card 1 at (1419, 650)
    time.sleep(3)
    player.place_card(2, 1419, 650)  # Place card 2 at (1419, 650)
    time.sleep(3)
    player.place_card(3, 1419, 650)  # Place card 3 at (1419, 650)
    time.sleep(3)

    # End match
    print("Ending match...")
    player.end_match()

def example_coordinate_capture():
    """Example of using coordinate capture"""
    print("\n=== Coordinate Capture Example ===")

    capturer = CoordinateCapture()

    print("Starting coordinate capture...")
    print("1. Click on your card slots (4 positions)")
    print("2. Click on start match button")
    print("3. Click on end match button")
    print("4. Press 'q' when done")

    capturer.start_capture()

    # Keep running until user quits
    try:
        while capturer.capture_mode:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user")

    capturer.stop_capture()

def example_card_strategy():
    """Example of placing all cards at the same location"""
    print("\n=== Card Strategy Example ===")

    player = GamePlayer()

    # Simple strategy: place cards in different lanes
    lanes = [
        (300, 600),  # Left lane
        (540, 700),  # Center lane
        (780, 600),  # Right lane
    ]

    print("Executing card placement strategy...")

    # Place all cards at the same location (1419, 650)
    target_location = (1419, 650)

    # Place cards 0-3 at the target location
    for card_index in range(4):
        print(f"Placing card {card_index} at ({target_location[0]}, {target_location[1]})")
        player.place_card(card_index, target_location[0], target_location[1])
        time.sleep(1.5)  # Wait between placements

if __name__ == "__main__":
    print("GamePlayer Examples")
    print("Choose an example to run:")
    print("1. Basic gameplay")
    print("2. Coordinate capture")
    print("3. Card strategy")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        example_basic_gameplay()
    elif choice == "2":
        example_coordinate_capture()
    elif choice == "3":
        example_card_strategy()
    else:
        print("Invalid choice")
