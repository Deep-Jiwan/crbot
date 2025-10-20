#!/usr/bin/env python3
"""
Simple coordinate testing script for GamePlayer module.

This script reads captured coordinates and allows you to test them
by clicking on the specified coordinates.
"""

import pyautogui
import time
import os
from pathlib import Path

def load_coordinates():
    """Load coordinates from captured_coordinates.txt file"""
    coord_file = Path(__file__).parent / "captured_coordinates.txt"

    if not coord_file.exists():
        print(f"❌ Coordinates file not found: {coord_file}")
        return []

    coordinates = []
    try:
        with open(coord_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and line.startswith('(') and line.endswith(')'):
                    # Parse coordinate like "(123, 456)"
                    try:
                        coord_str = line[1:-1]  # Remove parentheses
                        x, y = coord_str.split(',')
                        coordinates.append((int(x.strip()), int(y.strip())))
                    except ValueError as e:
                        print(f"⚠️  Skipping invalid coordinate on line {line_num}: {line}")
    except Exception as e:
        print(f"❌ Error reading coordinates file: {e}")
        return []

    return coordinates

def test_coordinate(x, y, index):
    """Test a specific coordinate by clicking on it"""
    print(f"\n🖱️  Testing coordinate {index + 1}: ({x}, {y})")
    print("Moving mouse and clicking in 3 seconds...")
    print("Make sure your game window is active!")

    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    try:
        # Move to coordinate and click
        pyautogui.moveTo(x, y, duration=0.5)
        time.sleep(0.5)
        pyautogui.click()
        print(f"✅ Successfully clicked at ({x}, {y})")
    except Exception as e:
        print(f"❌ Error clicking at ({x}, {y}): {e}")

def main():
    """Main coordinate testing function"""
    print("🎯 Coordinate Tester for GamePlayer")
    print("=" * 40)

    # Load coordinates
    coordinates = load_coordinates()

    if not coordinates:
        print("❌ No coordinates found. Make sure to run coordinate_capture.py first.")
        return

    print(f"\n📋 Found {len(coordinates)} coordinates:")
    for i, (x, y) in enumerate(coordinates):
        print(f"  {i + 1}. ({x}, {y})")

    while True:
        print("\n" + "=" * 40)
        print("Options:")
        print("1. Test a specific coordinate (enter number)")
        print("2. Test all coordinates")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "3":
            print("👋 Goodbye!")
            break

        elif choice == "2":
            print("\n🧪 Testing all coordinates...")
            for i, (x, y) in enumerate(coordinates):
                test_coordinate(x, y, i)
                if i < len(coordinates) - 1:  # Don't wait after last coordinate
                    input("Press Enter to test next coordinate...")

        elif choice == "1":
            try:
                coord_num = int(input(f"Enter coordinate number (1-{len(coordinates)}): "))
                if 1 <= coord_num <= len(coordinates):
                    x, y = coordinates[coord_num - 1]
                    test_coordinate(x, y, coord_num - 1)
                else:
                    print(f"❌ Invalid number. Please enter 1-{len(coordinates)}")
            except ValueError:
                print("❌ Please enter a valid number")

        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3")

if __name__ == "__main__":
    main()
