"""
Training Pipeline for Clash Royale AI

This module provides comprehensive training capabilities including:
- Data collection and preprocessing
- Model training with different strategies
- Evaluation and validation
- Hyperparameter tuning
- Model checkpointing and saving
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse
from pathlib import Path
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))
from clash_royale_ai import ClashRoyalePPO, ClashRoyaleDataset, GameState, Action

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training"""
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    sequence_length: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    early_stopping_patience: int = 10
    save_every_n_epochs: int = 10
    log_every_n_steps: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_save_dir: str = "models"
    log_dir: str = "logs"
    data_file: str = "../masterreceiver/game_data_log.jsonl"

class DataPreprocessor:
    """Preprocesses game data for training"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.feature_scalers = {}
        
    def load_and_preprocess_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess data from JSONL file"""
        logger.info(f"Loading data from {data_file}")
        
        sequences = []
        rewards = []
        
        with open(data_file, 'r') as f:
            current_sequence = []
            for line in f:
                try:
                    data = json.loads(line.strip())
                    current_sequence.append(data)
                    
                    if len(current_sequence) >= self.config.sequence_length:
                        # Extract features and reward
                        features = self._extract_features(current_sequence)
                        reward = self._calculate_reward(current_sequence[-1])
                        
                        sequences.append(features)
                        rewards.append(reward)
                        
                        # Sliding window
                        current_sequence.pop(0)
                        
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(sequences)} sequences")
        return np.array(sequences), np.array(rewards)
    
    def _extract_features(self, sequence: List[Dict]) -> np.ndarray:
        """Extract enhanced features from a sequence of game states"""
        features = []
        
        for state in sequence:
            # Basic features
            elixir = state.get('elixir', 0)
            cards_count = len(state.get('cards_in_hand', []))
            troops_count = len(state.get('troops', []))
            
            # Enhanced troop analysis
            troops = state.get('troops', [])
            enemy_troops = [t for t in troops if t.get('team') == 'enemy']
            ally_troops = [t for t in troops if t.get('team') == 'ally']
            
            # Enemy troop features
            enemy_count = len(enemy_troops)
            enemy_avg_x = np.mean([t.get('x', 0) for t in enemy_troops]) if enemy_troops else 0
            enemy_avg_y = np.mean([t.get('y', 0) for t in enemy_troops]) if enemy_troops else 0
            
            # Ally troop features  
            ally_count = len(ally_troops)
            ally_avg_x = np.mean([t.get('x', 0) for t in ally_troops]) if ally_troops else 0
            ally_avg_y = np.mean([t.get('y', 0) for t in ally_troops]) if ally_troops else 0
            
            # Win condition encoding
            win_condition = state.get('win_detection', 'ongoing')
            if win_condition == True:
                win_encoded = 1.0
            elif win_condition == False:
                win_encoded = -1.0
            else:
                win_encoded = 0.0
            
            # Card features
            cards = state.get('cards_in_hand', [])
            card_names = [card.get('name', '') for card in cards]
            unique_cards = len(set(card_names))
            
            # Troop balance
            troop_balance = ally_count - enemy_count
            
            # Distance between enemy and ally troops
            if enemy_troops and ally_troops:
                min_distance = min([
                    np.sqrt((e.get('x', 0) - a.get('x', 0))**2 + (e.get('y', 0) - a.get('y', 0))**2)
                    for e in enemy_troops for a in ally_troops
                ])
            else:
                min_distance = 1000.0  # Large distance if no troops
            
            # Time-based features
            timestamp = state.get('timestamp', 0)
            time_since_start = timestamp - sequence[0].get('timestamp', timestamp) if sequence else 0
            
            feature_vector = [
                elixir,
                cards_count,
                troops_count,
                enemy_count,
                ally_count,
                enemy_avg_x,
                enemy_avg_y,
                ally_avg_x,
                ally_avg_y,
                unique_cards,
                win_encoded,
                troop_balance,
                min_distance,
                len(cards),  # Redundant but kept for compatibility
                time_since_start
            ]
            
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_reward(self, state: Dict) -> float:
        """Calculate reward for a game state"""
        reward = 0.0
        
        # Elixir management reward
        elixir = state.get('elixir', 0)
        reward += elixir * 0.1
        
        # Card availability reward
        cards_count = len(state.get('cards_in_hand', []))
        reward += cards_count * 0.05
        
        # Win condition reward
        win_condition = state.get('win_detection', 'ongoing')
        if win_condition == True:
            reward += 10.0
        elif win_condition == False:
            reward -= 10.0
        
        # Troop balance reward
        troops = state.get('troops', [])
        enemy_troops = sum(1 for troop in troops if troop.get('team') == 'enemy')
        ally_troops = sum(1 for troop in troops if troop.get('team') == 'ally')
        
        if enemy_troops > ally_troops:
            reward -= 1.0  # Penalty for being outnumbered
        elif ally_troops > enemy_troops:
            reward += 1.0  # Bonus for having more troops
        
        return reward

class ModelTrainer:
    """Handles model training with various strategies"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create directories
        os.makedirs(config.model_save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(config.log_dir)
        
        # Initialize PPO model with enhanced input size (3 actions only)
        self.model = ClashRoyalePPO(input_size=15, hidden_size=256, num_actions=3).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_reward': [],
            'val_reward': []
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Train the model"""
        logger.info("Starting training...")
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss, train_reward = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_loss, val_reward = self._validate_epoch(val_loader, epoch)
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_reward'].append(train_reward)
            self.training_history['val_reward'].append(val_reward)
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}: "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"Train Reward: {train_reward:.4f}, Val Reward: {val_reward:.4f}")
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Reward/Train', train_reward, epoch)
            self.writer.add_scalar('Reward/Validation', val_reward, epoch)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Regular checkpointing
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)
        
        # Save final model
        self._save_checkpoint(self.config.epochs - 1, is_final=True)
        
        # Close tensorboard writer
        self.writer.close()
        
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0
        
        for batch_idx, (features, rewards) in enumerate(train_loader):
            features = features.to(self.device)
            rewards = rewards.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            predicted_values = outputs['value'].squeeze()
            
            # Calculate loss
            loss = self.criterion(predicted_values, rewards)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_reward += rewards.mean().item()
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.log_every_n_steps == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {loss.item():.4f}")
        
        return total_loss / num_batches, total_reward / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, rewards in val_loader:
                features = features.to(self.device)
                rewards = rewards.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                predicted_values = outputs['value'].squeeze()
                
                # Calculate loss
                loss = self.criterion(predicted_values, rewards)
                
                # Update metrics
                total_loss += loss.item()
                total_reward += rewards.mean().item()
                num_batches += 1
        
        return total_loss / num_batches, total_reward / num_batches
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.model_save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.model_save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch+1}")
        
        # Save final model
        if is_final:
            final_path = os.path.join(self.config.model_save_dir, 'final_model.pth')
            torch.save(checkpoint, final_path)
            logger.info(f"Saved final model at epoch {epoch+1}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")

class ModelEvaluator:
    """Evaluates trained models"""
    
    def __init__(self, model: ClashRoyalePPO, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set"""
        logger.info("Evaluating model...")
        
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for features, rewards in test_loader:
                features = features.to(self.device)
                rewards = rewards.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                predicted_values = outputs['value'].squeeze()
                
                # Calculate loss
                loss = nn.MSELoss()(predicted_values, rewards)
                total_loss += loss.item()
                
                # Store predictions and targets
                predictions.extend(predicted_values.cpu().numpy())
                targets.extend(rewards.cpu().numpy())
        
        # Calculate metrics
        mse = total_loss / len(test_loader)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
        
        # Calculate R-squared
        ss_res = np.sum((np.array(targets) - np.array(predictions)) ** 2)
        ss_tot = np.sum((np.array(targets) - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'targets': targets
        }
        
        logger.info(f"Evaluation metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, "
                   f"MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return metrics
    
    def plot_predictions(self, metrics: Dict, save_path: str = None):
        """Plot predictions vs targets"""
        predictions = metrics['predictions']
        targets = metrics['targets']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(targets, predictions, alpha=0.6)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', lw=2)
        plt.xlabel('Actual Rewards')
        plt.ylabel('Predicted Rewards')
        plt.title('Predictions vs Actual Rewards')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()

def create_data_loaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders"""
    # Load and preprocess data
    preprocessor = DataPreprocessor(config)
    features, rewards = preprocessor.load_and_preprocess_data(config.data_file)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(features),
        torch.FloatTensor(rewards)
    )
    
    # Split dataset
    total_size = len(dataset)
    test_size = int(total_size * config.test_split)
    val_size = int(total_size * config.validation_split)
    train_size = total_size - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    logger.info(f"Dataset split - Train: {len(train_dataset)}, "
               f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Clash Royale AI Model")
    parser.add_argument("--data-file", type=str, default="../masterreceiver/game_data_log.jsonl",
                       help="Path to training data file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model-save-dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate model")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to evaluate")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        model_save_dir=args.model_save_dir,
        log_dir=args.log_dir,
        data_file=args.data_file
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    if args.evaluate_only:
        # Evaluation only mode
        if not args.checkpoint:
            logger.error("Checkpoint path required for evaluation")
            return
        
        # Load PPO model with enhanced input size (3 actions only)
        model = ClashRoyalePPO(input_size=15, hidden_size=256, num_actions=3).to(config.device)
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        evaluator = ModelEvaluator(model, config.device)
        metrics = evaluator.evaluate(test_loader)
        
        # Plot results
        evaluator.plot_predictions(metrics, os.path.join(config.model_save_dir, 'evaluation_plot.png'))
        
    else:
        # Training mode
        trainer = ModelTrainer(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # Train model
        history = trainer.train(train_loader, val_loader)
        
        # Evaluate on test set
        evaluator = ModelEvaluator(trainer.model, config.device)
        test_metrics = evaluator.evaluate(test_loader)
        
        # Plot training history
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(history['train_reward'], label='Train Reward')
        plt.plot(history['val_reward'], label='Validation Reward')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.title('Training and Validation Reward')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        evaluator.plot_predictions(test_metrics)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.model_save_dir, 'training_history.png'))
        plt.show()
        
        logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()

