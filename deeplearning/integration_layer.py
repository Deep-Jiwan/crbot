"""
Integration Layer for Clash Royale AI

This module provides a unified interface to all existing modules and manages
the data flow between them and the deep learning model.
"""

import os
import sys
import time
import json
import threading
import subprocess
import signal
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import zmq
import numpy as np
import cv2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ModuleStatus:
    """Status of a module"""
    name: str
    running: bool = False
    process: Optional[subprocess.Popen] = None
    last_heartbeat: float = 0.0
    error_count: int = 0

class ModuleManager:
    """Manages all Clash Royale bot modules"""
    
    def __init__(self):
        self.modules = {}
        self.running = False
        self.setup_module_configs()
        
    def setup_module_configs(self):
        """Setup configuration for each module"""
        self.module_configs = {
            "publisher": {
                "path": "../publisher",
                "script": "publisher.py",
                "env_vars": {
                    "FRAME_WIDTH": "1080",
                    "FRAME_HEIGHT": "1920"
                },
                "required": True
            },
            "elixir_counter": {
                "path": "../elixircount", 
                "script": "elixir_count.py",
                "env_vars": {
                    "ZMQ_ADDRESS": "tcp://localhost:5550",
                    "PUB_PORT": "5560",
                    "ANNOTATE": "True"
                },
                "required": True
            },
            "card_detection": {
                "path": "../carddetection",
                "script": "card_detection.py", 
                "env_vars": {
                    "ZMQ_ADDRESS": "tcp://localhost:5550",
                    "PUB_PORT": "5590",
                    "ROBOFLOW_API_KEY": os.getenv("ROBOFLOW_API_KEY", ""),
                    "ROBOFLOW_WORKFLOW_ID": os.getenv("ROBOFLOW_WORKFLOW_ID", "")
                },
                "required": False
            },
            "troop_detection": {
                "path": "../troopdetection",
                "script": "troop_detection.py",
                "env_vars": {
                    "ZMQ_SUB_ADDRESS": "tcp://localhost:5550",
                    "PUB_PORT": "5580",
                    "ROBOFLOW_API_KEY": os.getenv("ROBOFLOW_API_KEY", ""),
                    "ROBOFLOW_WORKFLOW_ID": os.getenv("ROBOFLOW_WORKFLOW_ID", "")
                },
                "required": False
            },
            "win_detection": {
                "path": "../winwin",
                "script": "winwin.py",
                "env_vars": {
                    "ZMQ_ADDRESS": "tcp://localhost:5550",
                    "PUB_PORT": "5570"
                },
                "required": True
            },
            "master_receiver": {
                "path": "../masterreceiver",
                "script": "main.py",
                "env_vars": {
                    "TROOPS_PORT": "5580",
                    "ELIXIR_PORT": "5560", 
                    "WIN_PORT": "5570",
                    "CARDS_PORT": "5590"
                },
                "required": False
            }
        }
    
    def start_module(self, module_name: str) -> bool:
        """Start a specific module"""
        if module_name not in self.module_configs:
            print(f"Unknown module: {module_name}")
            return False
        
        config = self.module_configs[module_name]
        module_path = Path(__file__).parent / config["path"]
        script_path = module_path / config["script"]
        
        if not script_path.exists():
            print(f"Script not found: {script_path}")
            return False
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env.update(config["env_vars"])
            
            # Start the process
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(module_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Store module status
            self.modules[module_name] = ModuleStatus(
                name=module_name,
                running=True,
                process=process,
                last_heartbeat=time.time()
            )
            
            print(f"Started {module_name} (PID: {process.pid})")
            return True
            
        except Exception as e:
            print(f"Failed to start {module_name}: {e}")
            return False
    
    def stop_module(self, module_name: str) -> bool:
        """Stop a specific module"""
        if module_name not in self.modules:
            return False
        
        module = self.modules[module_name]
        if module.process:
            try:
                module.process.terminate()
                module.process.wait(timeout=5)
                print(f"Stopped {module_name}")
            except subprocess.TimeoutExpired:
                module.process.kill()
                print(f"Force killed {module_name}")
            except Exception as e:
                print(f"Error stopping {module_name}: {e}")
        
        module.running = False
        return True
    
    def start_all_modules(self, required_only: bool = False):
        """Start all modules"""
        print("Starting all modules...")
        
        for module_name, config in self.module_configs.items():
            if required_only and not config["required"]:
                continue
                
            if not self.start_module(module_name):
                if config["required"]:
                    print(f"Failed to start required module: {module_name}")
                    return False
        
        self.running = True
        print("All modules started successfully")
        return True
    
    def stop_all_modules(self):
        """Stop all modules"""
        print("Stopping all modules...")
        
        for module_name in list(self.modules.keys()):
            self.stop_module(module_name)
        
        self.running = False
        print("All modules stopped")
    
    def check_module_health(self):
        """Check health of all running modules"""
        current_time = time.time()
        
        for module_name, module in self.modules.items():
            if not module.running or not module.process:
                continue
            
            # Check if process is still running
            if module.process.poll() is not None:
                print(f"Module {module_name} has stopped unexpectedly")
                module.running = False
                module.error_count += 1
                
                # Restart if it's a required module
                config = self.module_configs[module_name]
                if config["required"] and module.error_count < 3:
                    print(f"Restarting {module_name}...")
                    self.start_module(module_name)
            else:
                module.last_heartbeat = current_time
                module.error_count = 0
    
    def get_module_status(self) -> Dict[str, Dict]:
        """Get status of all modules"""
        status = {}
        for name, module in self.modules.items():
            status[name] = {
                "running": module.running,
                "pid": module.process.pid if module.process else None,
                "error_count": module.error_count,
                "last_heartbeat": module.last_heartbeat
            }
        return status

class DataAggregator:
    """Aggregates data from all modules and provides unified interface"""
    
    def __init__(self):
        self.context = zmq.Context()
        self.setup_zmq()
        self.current_data = {
            "elixir": 0,
            "cards": [],
            "troops": [],
            "win_condition": "ongoing",
            "timestamp": 0
        }
        self.data_lock = threading.Lock()
        self.running = False
        
    def setup_zmq(self):
        """Setup ZeroMQ connections"""
        # Subscribe to all data sources
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect("tcp://localhost:5560")  # Elixir
        self.sub_socket.connect("tcp://localhost:5590")  # Cards  
        self.sub_socket.connect("tcp://localhost:5580")  # Troops
        self.sub_socket.connect("tcp://localhost:5570")  # Win detection
        
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"ecount|")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"cards|")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"troops|")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"winner|")
        
        # Set timeout for non-blocking receive
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
    
    def start(self):
        """Start data aggregation in background thread"""
        self.running = True
        self.thread = threading.Thread(target=self._aggregate_loop, daemon=True)
        self.thread.start()
        print("Data aggregator started")
    
    def stop(self):
        """Stop data aggregation"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
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
        """Get current aggregated data"""
        with self.data_lock:
            return self.current_data.copy()
    
    def cleanup(self):
        """Cleanup resources"""
        self.sub_socket.close()
        self.context.term()

class AIController:
    """Main controller that orchestrates all components"""
    
    def __init__(self):
        self.module_manager = ModuleManager()
        self.data_aggregator = DataAggregator()
        self.ai_agent = None
        self.running = False
        
    def initialize(self, start_modules: bool = True, required_only: bool = False):
        """Initialize the AI controller"""
        print("Initializing Clash Royale AI Controller...")
        
        # Start modules if requested
        if start_modules:
            if not self.module_manager.start_all_modules(required_only):
                print("Failed to start required modules")
                return False
        
        # Start data aggregation
        self.data_aggregator.start()
        
        # Initialize AI agent
        try:
            from clash_royale_ai import ClashRoyaleAgent
            self.ai_agent = ClashRoyaleAgent()
            print("AI agent initialized")
        except Exception as e:
            print(f"Failed to initialize AI agent: {e}")
            return False
        
        self.running = True
        print("AI Controller initialized successfully")
        return True
    
    def run(self, training: bool = True):
        """Run the AI controller"""
        if not self.running:
            print("Controller not initialized")
            return
        
        print("Starting AI Controller...")
        
        try:
            # Start health monitoring
            health_thread = threading.Thread(target=self._health_monitor, daemon=True)
            health_thread.start()
            
            # Run AI agent
            self.ai_agent.run(training)
            
        except KeyboardInterrupt:
            print("Shutting down AI Controller...")
        finally:
            self.shutdown()
    
    def _health_monitor(self):
        """Monitor health of all components"""
        while self.running:
            self.module_manager.check_module_health()
            time.sleep(5)  # Check every 5 seconds
    
    def shutdown(self):
        """Shutdown all components"""
        print("Shutting down AI Controller...")
        
        self.running = False
        
        # Stop AI agent
        if self.ai_agent:
            self.ai_agent.cleanup()
        
        # Stop data aggregation
        self.data_aggregator.stop()
        
        # Stop all modules
        self.module_manager.stop_all_modules()
        
        print("AI Controller shutdown complete")
    
    def get_status(self) -> Dict:
        """Get status of all components"""
        return {
            "running": self.running,
            "modules": self.module_manager.get_module_status(),
            "data": self.data_aggregator.get_current_data()
        }

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clash Royale AI Controller")
    parser.add_argument("--mode", choices=["train", "play"], default="play",
                       help="Mode: train the model or play the game")
    parser.add_argument("--required-only", action="store_true",
                       help="Start only required modules")
    parser.add_argument("--no-modules", action="store_true",
                       help="Don't start modules (assume they're already running)")
    
    args = parser.parse_args()
    
    # Create controller
    controller = AIController()
    
    # Initialize
    if not controller.initialize(not args.no_modules, args.required_only):
        print("Failed to initialize controller")
        return
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("Received interrupt signal")
        controller.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run controller
    try:
        controller.run(training=(args.mode == "train"))
    except Exception as e:
        print(f"Error running controller: {e}")
    finally:
        controller.shutdown()

if __name__ == "__main__":
    main()

