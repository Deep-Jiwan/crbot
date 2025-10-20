"""
Clash Royale Deep Learning AI Agent

This module implements a comprehensive deep learning model that integrates with all
existing modules to make intelligent decisions in Clash Royale gameplay.

Architecture:
- Multi-modal input processing (visual + game state)
- LSTM-based decision making with attention mechanism
- Reinforcement learning with experience replay
- Real-time action execution through GamePlayer integration
"""

import os
import sys
import time
import json
import threading
import numpy as np
import cv2
import zmq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Add parent directory to path to import gameplayer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gameplayer'))
from gameplayer import GamePlayer

# Load environment variables
load_dotenv()

@dataclass
class GameState:
    """Represents the current state of the game"""
    elixir: int = 0
    cards_in_hand: List[Dict] = None
    troops_on_field: List[Dict] = None
    win_condition: str = "ongoing"  # "ongoing", "win", "lose"
    timestamp: float = 0.0
    frame: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.cards_in_hand is None:
            self.cards_in_hand = []
        if self.troops_on_field is None:
            self.troops_on_field = []

@dataclass
class Action:
    """Represents an action the AI can take"""
    action_type: str  # "place_card", "wait", "start_match", "end_match"
    card_slot: Optional[int] = None  # 0-3 for card slots
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    confidence: float = 0.0

class ClashRoyaleDataset(Dataset):
    """Dataset for training the AI model"""
    
    def __init__(self, data_file: str, sequence_length: int = 10):
        self.data_file = data_file
        self.sequence_length = sequence_length
        self.sequences = self._load_sequences()
    
    def _load_sequences(self):
        """Load and preprocess game sequences from JSONL file"""
        sequences = []
        current_sequence = []
        
        with open(self.data_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    current_sequence.append(data)
                    
                    if len(current_sequence) >= self.sequence_length:
                        sequences.append(current_sequence.copy())
                        current_sequence.pop(0)  # Sliding window
                except json.JSONDecodeError:
                    continue
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Extract features
        elixir = torch.tensor([s.get('elixir', 0) for s in sequence], dtype=torch.float32)
        cards = torch.tensor([len(s.get('cards_in_hand', [])) for s in sequence], dtype=torch.float32)
        troops = torch.tensor([len(s.get('troops', [])) for s in sequence], dtype=torch.float32)
        
        # Combine features
        features = torch.stack([elixir, cards, troops], dim=1)
        
        # For now, use a simple reward based on win condition
        # In a real implementation, this would be more sophisticated
        reward = 0.0
        if sequence[-1].get('win_detection') == True:
            reward = 1.0
        elif sequence[-1].get('win_detection') == False:
            reward = -1.0
        
        return features, torch.tensor(reward, dtype=torch.float32)

class ClashRoyaleAI(nn.Module):
    """Main AI model for Clash Royale decision making"""
    
    def __init__(self, input_size: int = 3, hidden_size: int = 128, num_actions: int = 5):
        super(ClashRoyaleAI, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Action prediction heads
        self.action_head = nn.Linear(hidden_size, num_actions)
        self.card_head = nn.Linear(hidden_size, 4)  # 4 card slots
        self.position_head = nn.Linear(hidden_size, 2)  # x, y coordinates
        self.confidence_head = nn.Linear(hidden_size, 1)
        
        # Value function for reinforcement learning
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output for decision making
        last_output = attn_out[:, -1, :]  # (batch_size, hidden_size)
        last_output = self.dropout(last_output)
        
        # Predict actions
        action_logits = self.action_head(last_output)
        card_logits = self.card_head(last_output)
        position = self.position_head(last_output)
        confidence = torch.sigmoid(self.confidence_head(last_output))
        value = self.value_head(last_output)
        
        return {
            'action_logits': action_logits,
            'card_logits': card_logits,
            'position': position,
            'confidence': confidence,
            'value': value
        }

class ExperienceReplay:
    """Experience replay buffer for reinforcement learning"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class ClashRoyaleAgent:
    """Main agent that coordinates all components"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = ClashRoyaleAI().to(self.device)
        self.target_model = ClashRoyaleAI().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Load pretrained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.replay_buffer = ExperienceReplay()
        
        # Game state tracking
        self.current_state = GameState()
        self.state_history = deque(maxlen=100)
        
        # ZeroMQ setup
        self.setup_zmq()
        
        # Game player integration
        self.game_player = GamePlayer()
        
        # Action mapping
        self.action_map = {
            0: "wait",
            1: "place_card",
            2: "start_match", 
            3: "end_match",
            4: "defend"
        }
        
        # Training parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.update_target_freq = 100
        self.step_count = 0
        
    def setup_zmq(self):
        """Setup ZeroMQ connections to receive game data"""
        self.context = zmq.Context()
        
        # Subscribe to all game data
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect("tcp://localhost:5551")  # Elixir
        self.sub_socket.connect("tcp://localhost:5552")  # Cards
        self.sub_socket.connect("tcp://localhost:5560")  # Troops
        self.sub_socket.connect("tcp://localhost:5570")  # Win detection
        
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"ecount|")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"cards|")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"troops|")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"winner|")
        
        # Publisher for AI decisions
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:5580")
        
    def update_game_state(self, topic: str, data: str):
        """Update current game state based on received data"""
        if topic == "ecount":
            self.current_state.elixir = int(data)
        elif topic == "cards":
            self.current_state.cards_in_hand = self._parse_cards(data)
        elif topic == "troops":
            self.current_state.troops_on_field = self._parse_troops(data)
        elif topic == "winner":
            self.current_state.win_condition = data
        
        self.current_state.timestamp = time.time()
        
    def _parse_cards(self, cards_data: str) -> List[Dict]:
        """Parse card data from string format"""
        cards = []
        if cards_data:
            for card_info in cards_data.split(","):
                if ":" in card_info:
                    slot, name = card_info.split(":", 1)
                    cards.append({"slot": int(slot), "name": name})
        return cards
    
    def _parse_troops(self, troops_data: str) -> List[Dict]:
        """Parse troop data from JSON format"""
        try:
            troops_json = json.loads(troops_data)
            troops = []
            if isinstance(troops_json, list) and troops_json:
                for result in troops_json:
                    predictions = result.get("predictions", {}).get("predictions", [])
                    for pred in predictions:
                        troops.append({
                            "type": pred.get("class", "Unknown"),
                            "confidence": float(pred.get("confidence", 0)),
                            "x": float(pred.get("x", 0)),
                            "y": float(pred.get("y", 0)),
                            "team": "enemy" if "enemy" in pred.get("class", "").lower() else "ally"
                        })
            return troops
        except json.JSONDecodeError:
            return []
    
    def get_state_features(self) -> np.ndarray:
        """Convert current game state to feature vector"""
        features = np.array([
            self.current_state.elixir,
            len(self.current_state.cards_in_hand),
            len(self.current_state.troops_on_field)
        ], dtype=np.float32)
        return features
    
    def select_action(self, state_features: np.ndarray, training: bool = True) -> Action:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action for exploration
            action_type = random.choice(list(self.action_map.values()))
            card_slot = random.randint(0, 3) if action_type == "place_card" else None
            target_x = random.randint(200, 880) if action_type == "place_card" else None
            target_y = random.randint(200, 1200) if action_type == "place_card" else None
            return Action(action_type, card_slot, target_x, target_y, 0.5)
        
        # Use model for action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0).unsqueeze(0).to(self.device)
            outputs = self.model(state_tensor)
            
            action_logits = outputs['action_logits']
            card_logits = outputs['card_logits']
            position = outputs['position']
            confidence = outputs['confidence']
            
            # Select action
            action_idx = torch.argmax(action_logits).item()
            action_type = self.action_map[action_idx]
            
            # Select card and position if placing a card
            card_slot = None
            target_x = None
            target_y = None
            
            if action_type == "place_card":
                card_slot = torch.argmax(card_logits).item()
                # Convert normalized position to actual coordinates
                target_x = int(position[0, 0].item() * 680 + 200)  # Scale to game area
                target_y = int(position[0, 1].item() * 1000 + 200)
            
            conf_value = confidence[0, 0].item()
            
            return Action(action_type, card_slot, target_x, target_y, conf_value)
    
    def execute_action(self, action: Action):
        """Execute the selected action using GamePlayer"""
        try:
            if action.action_type == "place_card" and action.card_slot is not None:
                if action.target_x and action.target_y:
                    self.game_player.place_card(action.card_slot, action.target_x, action.target_y)
                    print(f"AI placed card {action.card_slot} at ({action.target_x}, {action.target_y})")
            elif action.action_type == "start_match":
                self.game_player.start_match()
                print("AI started match")
            elif action.action_type == "end_match":
                self.game_player.end_match()
                print("AI ended match")
            elif action.action_type == "wait":
                time.sleep(0.5)  # Wait briefly
                print("AI waiting...")
        except Exception as e:
            print(f"Error executing action: {e}")
    
    def train_step(self):
        """Perform one training step using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_outputs = self.model(states)
        current_values = current_outputs['value'].squeeze()
        
        # Next Q values from target network
        with torch.no_grad():
            next_outputs = self.target_model(next_states)
            next_values = next_outputs['value'].squeeze()
            target_values = rewards + (0.99 * next_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_values, target_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def run(self, training: bool = True):
        """Main game loop"""
        print("Starting Clash Royale AI Agent...")
        
        last_action_time = 0
        action_interval = 2.0  # Minimum time between actions
        
        try:
            while True:
                # Check for new messages
                try:
                    msg = self.sub_socket.recv(zmq.NOBLOCK)
                    topic, data = msg.decode().split("|", 1)
                    self.update_game_state(topic, data)
                except zmq.Again:
                    pass
                
                current_time = time.time()
                
                # Make decision if enough time has passed
                if current_time - last_action_time >= action_interval:
                    state_features = self.get_state_features()
                    action = self.select_action(state_features, training)
                    
                    # Execute action
                    self.execute_action(action)
                    
                    # Store experience for training
                    if training:
                        reward = self._calculate_reward()
                        self.replay_buffer.push(
                            state_features, action, reward, 
                            self.get_state_features(), False
                        )
                        
                        # Train model
                        self.train_step()
                    
                    last_action_time = current_time
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            print("Shutting down AI agent...")
        finally:
            self.cleanup()
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current game state"""
        reward = 0.0
        
        # Reward for having elixir (encourages resource management)
        reward += self.current_state.elixir * 0.1
        
        # Reward for having cards in hand
        reward += len(self.current_state.cards_in_hand) * 0.05
        
        # Reward for win condition
        if self.current_state.win_condition == "win":
            reward += 10.0
        elif self.current_state.win_condition == "lose":
            reward -= 10.0
        
        return reward
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.step_count = checkpoint.get('step_count', 0)
        print(f"Model loaded from {path}")
    
    def cleanup(self):
        """Clean up resources"""
        self.sub_socket.close()
        self.pub_socket.close()
        self.context.term()

def train_model(data_file: str, epochs: int = 100):
    """Train the model using historical data"""
    print(f"Training model with data from {data_file}")
    
    # Create dataset
    dataset = ClashRoyaleDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = ClashRoyaleAI().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (features, rewards) in enumerate(dataloader):
            optimizer.zero_grad()
            
            outputs = model(features)
            predicted_values = outputs['value'].squeeze()
            
            loss = criterion(predicted_values, rewards)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clash Royale AI Agent")
    parser.add_argument("--mode", choices=["train", "play"], default="play", 
                       help="Mode: train the model or play the game")
    parser.add_argument("--model-path", type=str, help="Path to saved model")
    parser.add_argument("--data-file", type=str, default="../masterreceiver/game_data_log.jsonl",
                       help="Path to training data file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model(args.data_file, args.epochs)
    else:
        agent = ClashRoyaleAgent(args.model_path)
        agent.run(training=True)
