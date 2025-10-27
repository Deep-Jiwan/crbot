import pyautogui
import time
import os
import random
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
            (int(os.getenv("CARD_0_X", "1489")), int(os.getenv("CARD_0_Y", "900"))),  # Card 0 (leftmost)
            (int(os.getenv("CARD_1_X", "1604")), int(os.getenv("CARD_1_Y", "900"))),  # Card 1
            (int(os.getenv("CARD_2_X", "1704")), int(os.getenv("CARD_2_Y", "900"))),  # Card 2
            (int(os.getenv("CARD_3_X", "1795")), int(os.getenv("CARD_3_Y", "900"))),  # Card 3 (rightmost)
        ]

        # Match control coordinates
        self.start_match_coords = (int(os.getenv("START_MATCH_X", "1603")), int(os.getenv("START_MATCH_Y", "1200")))
        self.end_match_coords = (int(os.getenv("END_MATCH_X", "1606")), int(os.getenv("END_MATCH_Y", "931")))

        # Arena boundaries
        self.arena_min_x = int(os.getenv("ARENA_MIN_X", "1380"))
        self.arena_min_y = int(os.getenv("ARENA_MIN_Y", "140"))
        self.arena_max_x = int(os.getenv("ARENA_MAX_X", "1834"))
        self.arena_max_y = int(os.getenv("ARENA_MAX_Y", "757"))

        # Delay settings
        self.click_delay = float(os.getenv("CLICK_DELAY", "0.5"))  # Delay between card selection and placement
        self.card_select_delay = float(os.getenv("CARD_SELECT_DELAY", "0.5"))  # Delay when selecting cards

        print("GamePlayer initialized with coordinates:")
        print(f"Card slots: {self.card_slots}")
        print(f"Start match: {self.start_match_coords}")
        print(f"End match: {self.end_match_coords}")
        print(f"Arena bounds: ({self.arena_min_x}, {self.arena_min_y}) to ({self.arena_max_x}, {self.arena_max_y})")
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

    def adjust_coords(self, x, y):
        """
        Adjust coordinates to be within arena boundaries.
        Clamps values to valid range to ensure cards are placed inside the arena.
        
        Args:
            x (int): X coordinate to adjust
            y (int): Y coordinate to adjust
            
        Returns:
            tuple: (adjusted_x, adjusted_y) within arena bounds
        """
        adjusted_x = max(self.arena_min_x, min(x, self.arena_max_x))
        adjusted_y = max(self.arena_min_y, min(y, self.arena_max_y))
        
        if adjusted_x != x or adjusted_y != y:
            print(f"Adjusted coords from ({x}, {y}) to ({adjusted_x}, {adjusted_y})")
        
        return adjusted_x, adjusted_y

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
            # Adjust coordinates if outside arena bounds
            if (location_x < self.arena_min_x or location_x > self.arena_max_x or 
                location_y < self.arena_min_y or location_y > self.arena_max_y):
                location_x, location_y = self.adjust_coords(location_x, location_y)
            
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

    def test_functionality(self):
        """Test GamePlayer functionality"""
        print("\n" + "="*50)
        print("TESTING GAMEPLAYER FUNCTIONALITY")
        print("="*50)
        
        # 1. Start match
        print("\n1. Starting match...")
        self.start_match()
        time.sleep(1)
        
        # 2. Select cards 0, 1, 2, 3 (just click, don't deploy)
        print("\n2. Selecting cards (1 second interval)...")
        for card in range(4):
            card_x, card_y = self.card_slots[card]
            print(f"   Clicking card {card} at ({card_x}, {card_y})")
            pyautogui.click(card_x, card_y)
            time.sleep(1)
        
        # 3. Deploy cards - Test boundary checking
        print("\n3. Deploying cards with boundary tests (1 second interval)...")
        
        test_positions = [
            (1500, 400, "INSIDE - Valid center position"),
            (1600, 600, "INSIDE - Valid lower position"),
            (2000, 50, "OUTSIDE - X beyond right edge (2000>1834), Y above top edge (50<140)"),
            (1000, 1000, "OUTSIDE - X beyond left edge (1000<1380), Y below bottom edge (1000>757)"),
        ]
        
        for card, (x, y, description) in enumerate(test_positions):
            print(f"\n   Test {card + 1}: {description}")
            print(f"   Requested position: ({x}, {y})")
            self.place_card(card, x, y)
            time.sleep(1)
        
        # 4. End match
        print("\n4. Ending match...")
        self.end_match()
        time.sleep(1)
        
        print("\n" + "="*50)
        print("TEST COMPLETE")
        print("="*50)
        print("\nTest Summary:")
        print("  - 2 positions inside arena bounds")
        print("  - 2 positions outside arena bounds (should be adjusted)")
        print("="*50)


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
