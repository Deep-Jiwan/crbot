"""
Data Aggregator Module for Clash Royale AI

This module aggregates data from all bot modules (elixir counter, card detection,
troop detection, win detection) via ZeroMQ and provides a unified interface for
accessing the current game state.
"""

import os
import time
import json
import threading
from typing import Dict, List
from pathlib import Path
import zmq


class DataAggregator:
    """Aggregates data from all modules and provides unified interface"""
    
    def __init__(self, elixir_port: int = 5560, cards_port: int = 5590,
                 troops_port: int = 5580, win_port: int = 5570,
                 timeout_ms: int = 100):
        """
        Initialize the DataAggregator.
        
        Args:
            elixir_port: Port for elixir count data
            cards_port: Port for card detection data
            troops_port: Port for troop detection data
            win_port: Port for win detection data
            timeout_ms: Timeout for ZMQ socket receive in milliseconds
        """
        self.elixir_port = elixir_port
        self.cards_port = cards_port
        self.troops_port = troops_port
        self.win_port = win_port
        self.timeout_ms = timeout_ms
        
        self.context = zmq.Context()
        self.sub_socket = None
        
        self.current_data = {
            "elixir": 0,
            "cards": [],
            "troops": [],
            "win_condition": "ongoing",
            "timestamp": 0
        }
        self.data_lock = threading.Lock()
        self.running = False
        self.thread = None
        
        # Logging setup
        self.enable_logging = os.getenv("LOG_GAME", "true").lower() != "false"
        self.log_frequency = float(os.getenv("LOG_FREQUENCY", "0"))  # 0 = log every update
        self.last_log_time = 0
        self.log_file = None
        if self.enable_logging:
            self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging directory and file"""
        # Get the deeplearning directory (parent of utils)
        deeplearning_dir = Path(__file__).parent.parent
        log_dir = deeplearning_dir / "game_logs"
        
        # Create directory if it doesn't exist
        log_dir.mkdir(exist_ok=True)
        
        # Open log file in append mode
        log_path = log_dir / "game_log.jsonl"
        self.log_file = open(log_path, 'a', encoding='utf-8')
        print(f"Logging enabled: {log_path}")
    
    def _log_data(self, data: Dict):
        """Log data to JSONL file based on frequency setting"""
        if not self.enable_logging or not self.log_file:
            return
        
        current_time = time.time()
        
        # Check if enough time has passed since last log
        if self.log_frequency > 0:
            if current_time - self.last_log_time < self.log_frequency:
                return  # Skip this log
        
        # Log the data
        try:
            json_line = json.dumps(data, ensure_ascii=False)
            self.log_file.write(json_line + '\n')
            self.log_file.flush()
            self.last_log_time = current_time
        except Exception as e:
            print(f"Error logging data: {e}")
        
    def setup_zmq(self):
        """Setup ZeroMQ connections"""
        # Subscribe to all data sources
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://localhost:{self.elixir_port}")
        self.sub_socket.connect(f"tcp://localhost:{self.cards_port}")
        self.sub_socket.connect(f"tcp://localhost:{self.troops_port}")
        self.sub_socket.connect(f"tcp://localhost:{self.win_port}")
        
        # Subscribe to specific topics
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"ecount|")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"cards|")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"troops|")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"winner|")
        
        # Set timeout for non-blocking receive
        self.sub_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
    
    def start(self):
        """Start data aggregation in background thread"""
        if self.running:
            print("DataAggregator is already running")
            return
        
        self.setup_zmq()
        self.running = True
        self.thread = threading.Thread(target=self._aggregate_loop, daemon=True)
        self.thread.start()
        print("Data aggregator started")
    
    def stop(self):
        """Stop data aggregation"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("Data aggregator stopped")
    
    def _aggregate_loop(self):
        """Main aggregation loop"""
        while self.running:
            try:
                msg = self.sub_socket.recv()
                topic, data = msg.decode().split("|", 1)
                self._process_message(topic, data)
            except zmq.Again:
                # Timeout - continue loop
                continue
            except Exception as e:
                print(f"Error in data aggregation: {e}")
                time.sleep(0.1)
    
    def _process_message(self, topic: str, data: str):
        """Process incoming message and update current data"""
        with self.data_lock:
            if topic == "ecount":
                self.current_data["elixir"] = int(data)
            elif topic == "cards":
                self.current_data["cards"] = self._parse_cards(data)
            elif topic == "troops":
                self.current_data["troops"] = self._parse_troops(data)
            elif topic == "winner":
                self.current_data["win_condition"] = data
            
            self.current_data["timestamp"] = time.time()
            
            # Log the updated data
            self._log_data(self.current_data.copy())
    
    def _parse_cards(self, cards_data: str) -> List[Dict]:
        """Parse card data"""
        cards = []
        if cards_data:
            for card_info in cards_data.split(","):
                if ":" in card_info:
                    slot, name = card_info.split(":", 1)
                    cards.append({"slot": int(slot), "name": name})
        return cards
    
    def _parse_troops(self, troops_data: str) -> List[Dict]:
        """Parse troop data with enhanced team detection"""
        try:
            troops_json = json.loads(troops_data)
            troops = []
            if isinstance(troops_json, list) and troops_json:
                for result in troops_json:
                    predictions = result.get("predictions", {}).get("predictions", [])
                    for pred in predictions:
                        # Enhanced team detection
                        class_name = pred.get("class", "").lower()
                        if "enemy" in class_name or "opponent" in class_name:
                            team = "enemy"
                        elif "ally" in class_name or "friendly" in class_name or "player" in class_name:
                            team = "ally"
                        else:
                            # Default team assignment based on position
                            y_pos = float(pred.get("y", 0))
                            team = "enemy" if y_pos < 960 else "ally"  # Assuming 1920 height, enemy is top half
                        
                        troops.append({
                            "type": pred.get("class", "Unknown"),
                            "confidence": float(pred.get("confidence", 0)),
                            "x": float(pred.get("x", 0)),
                            "y": float(pred.get("y", 0)),
                            "team": team
                        })
            return troops
        except json.JSONDecodeError:
            return []
    
    def get_current_data(self) -> Dict:
        """Get current aggregated data as JSON-serializable dict"""
        with self.data_lock:
            return self.current_data.copy()
    
    def get_current_data_json(self) -> str:
        """Get current aggregated data as JSON string"""
        with self.data_lock:
            return json.dumps(self.current_data, ensure_ascii=False)
    
    def get_elixir(self) -> int:
        """Get current elixir count"""
        with self.data_lock:
            return self.current_data["elixir"]
    
    def get_cards(self) -> List[Dict]:
        """Get current available cards"""
        with self.data_lock:
            return self.current_data["cards"].copy()
    
    def get_troops(self) -> List[Dict]:
        """Get current troops on battlefield"""
        with self.data_lock:
            return self.current_data["troops"].copy()
    
    def get_win_condition(self) -> str:
        """Get current win condition status"""
        with self.data_lock:
            return self.current_data["win_condition"]
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        if self.sub_socket:
            self.sub_socket.close()
        self.context.term()
        
        # Close log file
        if self.log_file:
            self.log_file.close()
            print("Log file closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


if __name__ == "__main__":
    """Example usage of DataAggregator"""
    print("Starting DataAggregator example...")
    
    # Create and start aggregator
    aggregator = DataAggregator()
    aggregator.start()
    
    try:
        # Monitor data for 30 seconds
        for i in range(30):
            time.sleep(1)
            data = aggregator.get_current_data()
            print(f"\n--- Update {i+1} ---")
            print(f"Elixir: {data['elixir']}")
            print(f"Cards: {len(data['cards'])}")
            print(f"Troops: {len(data['troops'])}")
            print(f"Win Condition: {data['win_condition']}")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        aggregator.cleanup()
