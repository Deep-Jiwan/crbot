#!/usr/bin/env python3
"""
Main Controller for Clash Royale Deep Learning AI

This is the main entry point that orchestrates all components of the Clash Royale AI system.
It provides a unified interface to start, stop, and monitor all modules and the AI agent.

Usage:
    python main.py --mode play                    # Play mode with AI
    python main.py --mode train                   # Training mode
    python main.py --mode collect-data           # Data collection mode
    python main.py --mode evaluate               # Evaluation mode
    python main.py --mode status                 # Check system status
"""

import os
import sys
import time
import json
import argparse
import signal
import threading
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from integration_layer import AIController
from clash_royale_ai import ClashRoyalePPOAgent, ClashRoyalePPO
from training_pipeline import TrainingConfig, ModelTrainer, ModelEvaluator, create_data_loaders

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_controller.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClashRoyaleAIManager:
    """Main manager for the Clash Royale AI system"""
    
    def __init__(self):
        self.controller = None
        self.running = False
        self.mode = None
        
    def initialize(self, mode: str, **kwargs):
        """Initialize the AI manager with specified mode"""
        self.mode = mode
        logger.info(f"Initializing Clash Royale AI Manager in {mode} mode")
        
        if mode == "play":
            self._initialize_play_mode(**kwargs)
        elif mode == "train":
            self._initialize_train_mode(**kwargs)
        elif mode == "collect-data":
            self._initialize_data_collection_mode(**kwargs)
        elif mode == "evaluate":
            self._initialize_evaluate_mode(**kwargs)
        elif mode == "status":
            self._initialize_status_mode(**kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _initialize_play_mode(self, model_path: Optional[str] = None, 
                            required_only: bool = False, no_modules: bool = False):
        """Initialize for play mode"""
        self.controller = AIController()
        
        if not self.controller.initialize(start_modules=not no_modules, required_only=required_only):
            raise RuntimeError("Failed to initialize AI controller")
        
        self.running = True
        logger.info("Play mode initialized successfully")
    
    def _initialize_train_mode(self, data_file: str, epochs: int = 100, 
                             batch_size: int = 32, learning_rate: float = 0.001):
        """Initialize for training mode"""
        self.training_config = TrainingConfig(
            data_file=data_file,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        self.running = True
        logger.info("Training mode initialized successfully")
    
    def _initialize_data_collection_mode(self, required_only: bool = False):
        """Initialize for data collection mode"""
        self.controller = AIController()
        
        if not self.controller.initialize(start_modules=True, required_only=required_only):
            raise RuntimeError("Failed to initialize data collection")
        
        self.running = True
        logger.info("Data collection mode initialized successfully")
    
    def _initialize_evaluate_mode(self, model_path: str, data_file: str):
        """Initialize for evaluation mode"""
        self.model_path = model_path
        self.data_file = data_file
        self.running = True
        logger.info("Evaluation mode initialized successfully")
    
    def _initialize_status_mode(self):
        """Initialize for status checking mode"""
        self.running = True
        logger.info("Status mode initialized successfully")
    
    def run(self):
        """Run the AI manager based on current mode"""
        if not self.running:
            logger.error("Manager not initialized")
            return
        
        try:
            if self.mode == "play":
                self._run_play_mode()
            elif self.mode == "train":
                self._run_train_mode()
            elif self.mode == "collect-data":
                self._run_data_collection_mode()
            elif self.mode == "evaluate":
                self._run_evaluate_mode()
            elif self.mode == "status":
                self._run_status_mode()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in {self.mode} mode: {e}")
        finally:
            self.shutdown()
    
    def _run_play_mode(self):
        """Run play mode"""
        logger.info("Starting play mode...")
        self.controller.run(training=False)
    
    def _run_train_mode(self):
        """Run training mode"""
        logger.info("Starting training mode...")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(self.training_config)
        
        # Create trainer
        trainer = ModelTrainer(self.training_config)
        
        # Train model
        history = trainer.train(train_loader, val_loader)
        
        # Evaluate on test set
        evaluator = ModelEvaluator(trainer.model, trainer.device)
        test_metrics = evaluator.evaluate(test_loader)
        
        logger.info("Training completed successfully!")
    
    def _run_data_collection_mode(self):
        """Run data collection mode"""
        logger.info("Starting data collection mode...")
        logger.info("Data will be collected and saved to masterreceiver/game_data_log.jsonl")
        logger.info("Press Ctrl+C to stop data collection")
        
        try:
            while True:
                status = self.controller.get_status()
                logger.info(f"Data collection status: {status}")
                time.sleep(10)  # Log status every 10 seconds
        except KeyboardInterrupt:
            logger.info("Data collection stopped")
    
    def _run_evaluate_mode(self):
        """Run evaluation mode"""
        logger.info("Starting evaluation mode...")
        
        # Load PPO model
        import torch
        model = ClashRoyalePPO(input_size=15, hidden_size=256)
        checkpoint = torch.load(self.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create data loaders
        config = TrainingConfig(data_file=self.data_file)
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        # Evaluate
        evaluator = ModelEvaluator(model, torch.device('cpu'))
        metrics = evaluator.evaluate(test_loader)
        
        # Print results
        print(f"Evaluation Results:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
    
    def _run_status_mode(self):
        """Run status checking mode"""
        logger.info("Checking system status...")
        
        # Check if data file exists
        data_file = Path("../masterreceiver/game_data_log.jsonl")
        if data_file.exists():
            file_size = data_file.stat().st_size
            logger.info(f"Data file exists: {data_file} ({file_size} bytes)")
        else:
            logger.warning(f"Data file not found: {data_file}")
        
        # Check if models directory exists
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pth"))
            logger.info(f"Models directory exists with {len(model_files)} model files")
            for model_file in model_files:
                logger.info(f"  - {model_file.name}")
        else:
            logger.warning("Models directory not found")
        
        # Check if required modules can be imported
        try:
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
        except ImportError:
            logger.error("PyTorch not installed")
        
        try:
            import cv2
            logger.info(f"OpenCV version: {cv2.__version__}")
        except ImportError:
            logger.error("OpenCV not installed")
        
        try:
            import zmq
            logger.info(f"PyZMQ version: {zmq.zmq_version()}")
        except ImportError:
            logger.error("PyZMQ not installed")
    
    def shutdown(self):
        """Shutdown the AI manager"""
        if self.controller:
            self.controller.shutdown()
        
        self.running = False
        logger.info("AI Manager shutdown complete")

def setup_signal_handlers(manager: ClashRoyaleAIManager):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        manager.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Clash Royale Deep Learning AI Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Play mode with AI (requires trained model)
  python main.py --mode play --model-path models/best_model.pth
  
  # Training mode
  python main.py --mode train --data-file ../masterreceiver/game_data_log.jsonl --epochs 100
  
  # Data collection mode (just run modules to collect data)
  python main.py --mode collect-data --required-only
  
  # Evaluate trained model
  python main.py --mode evaluate --model-path models/best_model.pth --data-file ../masterreceiver/game_data_log.jsonl
  
  # Check system status
  python main.py --mode status
        """
    )
    
    parser.add_argument("--mode", 
                       choices=["play", "train", "collect-data", "evaluate", "status"],
                       required=True,
                       help="Mode to run the AI system")
    
    # Play mode arguments
    parser.add_argument("--model-path", type=str,
                       help="Path to trained model for play mode")
    parser.add_argument("--required-only", action="store_true",
                       help="Start only required modules")
    parser.add_argument("--no-modules", action="store_true",
                       help="Don't start modules (assume they're already running)")
    
    # Training mode arguments
    parser.add_argument("--data-file", type=str, default="../masterreceiver/game_data_log.jsonl",
                       help="Path to training data file")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate for training")
    
    args = parser.parse_args()
    
    # Create AI manager
    manager = ClashRoyaleAIManager()
    
    # Setup signal handlers
    setup_signal_handlers(manager)
    
    try:
        # Initialize based on mode
        if args.mode == "play":
            manager.initialize("play", 
                             model_path=args.model_path,
                             required_only=args.required_only,
                             no_modules=args.no_modules)
        elif args.mode == "train":
            manager.initialize("train",
                             data_file=args.data_file,
                             epochs=args.epochs,
                             batch_size=args.batch_size,
                             learning_rate=args.learning_rate)
        elif args.mode == "collect-data":
            manager.initialize("collect-data",
                             required_only=args.required_only)
        elif args.mode == "evaluate":
            if not args.model_path:
                logger.error("Model path required for evaluation mode")
                return
            manager.initialize("evaluate",
                             model_path=args.model_path,
                             data_file=args.data_file)
        elif args.mode == "status":
            manager.initialize("status")
        
        # Run the manager
        manager.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
