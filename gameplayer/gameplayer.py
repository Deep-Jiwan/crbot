import pyautogui
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GamePlayer:
    """
    A class to automate Clash Royale gameplay through mouse clicks.

    Core functions:
    - Start matches by clicking specific coordinates
    - Place cards by clicking card slots then target locations
    - End matches by clicking specific coordinates
    """

    def __init__(self):
        # Card slot coordinates (x, y) for cards 0-3 (left to right)
        self.card_slots = [
            (int(os.getenv("CARD_0_X", "225")), int(os.getenv("CARD_0_Y", "1560"))),  # Card 0 (leftmost)
            (int(os.getenv("CARD_1_X", "415")), int(os.getenv("CARD_1_Y", "1560"))),  # Card 1
            (int(os.getenv("CARD_2_X", "605")), int(os.getenv("CARD_2_Y", "1560"))),  # Card 2
            (int(os.getenv("CARD_3_X", "795")), int(os.getenv("CARD_3_Y", "1560"))),  # Card 3 (rightmost)
        ]

        # Match control coordinates
        self.start_match_coords = (int(os.getenv("START_MATCH_X", "540")), int(os.getenv("START_MATCH_Y", "1200")))
        self.end_match_coords = (int(os.getenv("END_MATCH_X", "540")), int(os.getenv("END_MATCH_Y", "1400")))

        # Delay settings
        self.click_delay = float(os.getenv("CLICK_DELAY", "0.1"))  # Delay between card selection and placement
        self.card_select_delay = float(os.getenv("CARD_SELECT_DELAY", "0.5"))  # Delay when selecting cards

        print("GamePlayer initialized with coordinates:")
        print(f"Card slots: {self.card_slots}")
        print(f"Start match: {self.start_match_coords}")
        print(f"End match: {self.end_match_coords}")

    def start_match(self):
        """
        Click on the start match button to begin a game.
        """
        try:
            print(f"Starting match by clicking at {self.start_match_coords}")
            pyautogui.click(self.start_match_coords[0], self.start_match_coords[1])
            time.sleep(1)  # Wait for match to start
            print("Match start command sent")
        except Exception as e:
            print(f"Error starting match: {e}")

    def place_card(self, card_no, location_x, location_y):
        """
        Place a card by first clicking on the card slot, then clicking on the target location.

        Args:
            card_no (int): Card number (0-3, left to right)
            location_x (int): X coordinate where to place the card
            location_y (int): Y coordinate where to place the card
        """
        if not (0 <= card_no <= 3):
            print(f"Invalid card number: {card_no}. Must be 0-3.")
            return

        try:
            card_x, card_y = self.card_slots[card_no]
            print(f"Placing card {card_no} from ({card_x}, {card_y}) to ({location_x}, {location_y})")

            # Step 1: Click on the card to select it
            pyautogui.click(card_x, card_y)
            time.sleep(self.card_select_delay)

            # Step 2: Click on the target location to place it
            pyautogui.click(location_x, location_y)
            time.sleep(self.click_delay)

            print(f"Card {card_no} placed successfully")

        except Exception as e:
            print(f"Error placing card: {e}")

    def end_match(self):
        """
        Click on the end match/surrender button.
        """
        try:
            print(f"Ending match by clicking at {self.end_match_coords}")
            pyautogui.click(self.end_match_coords[0], self.end_match_coords[1])
            time.sleep(1)  # Wait for action to complete
            print("Match end command sent")
        except Exception as e:
            print(f"Error ending match: {e}")

    def set_card_slot(self, card_no, x, y):
        """Update card slot coordinates for a specific card"""
        if 0 <= card_no <= 3:
            self.card_slots[card_no] = (x, y)
            print(f"Updated card {card_no} coordinates to ({x}, {y})")

    def set_start_match_coords(self, x, y):
        """Update start match button coordinates"""
        self.start_match_coords = (x, y)
        print(f"Updated start match coordinates to ({x}, {y})")

    def set_end_match_coords(self, x, y):
        """Update end match button coordinates"""
        self.end_match_coords = (x, y)
        print(f"Updated end match coordinates to ({x}, {y})")


# Example usage
if __name__ == "__main__":
    player = GamePlayer()

    print("\nGamePlayer loaded successfully!")
    print("Available functions:")
    print("- start_match()")
    print("- place_card(card_no, x, y)")
    print("- end_match()")
    print("- set_card_slot(card_no, x, y)")
    print("- set_start_match_coords(x, y)")
    print("- set_end_match_coords(x, y)")

    # Uncomment to test functions:
    # player.start_match()
    # player.place_card(0, 1419, 650)  # Place card 0 at coordinates (1419, 650)
    # player.end_match()
