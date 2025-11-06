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
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import logging
from datetime import datetime

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
    enemy_troops: List[Dict] = None
    ally_troops: List[Dict] = None
    
    def __post_init__(self):
        if self.cards_in_hand is None:
            self.cards_in_hand = []
        if self.troops_on_field is None:
            self.troops_on_field = []
        if self.enemy_troops is None:
            self.enemy_troops = []
        if self.ally_troops is None:
            self.ally_troops = []

@dataclass
class Action:
    """Represents an action the AI can take"""
    action_type: str  # "place_card", "wait", "start_match", "end_match"
    card_slot: Optional[int] = None  # 0-3 for card slots
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    target_zone: Optional[str] = None  # "bottom_left", "bottom_center", "bottom_right", "top_left", "top_center", "top_right"
    confidence: float = 0.0
    reasoning: str = ""  # Human-readable reasoning for the decision
    card_name: Optional[str] = None  # Name of the card being played
    timestamp: float = 0.0

class DecisionLogger:
    """Logs AI decisions in structured JSON format"""
    
    def __init__(self, log_file: str = "ai_decisions.jsonl"):
        self.log_file = log_file
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ai_decisions.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_decision(self, game_state: GameState, action: Action, model_outputs: Dict):
        """Log a complete AI decision with context"""
        decision_log = {
            "timestamp": datetime.now().isoformat(),
            "game_state": {
                "elixir": game_state.elixir,
                "cards_in_hand": game_state.cards_in_hand,
                "enemy_troops": game_state.enemy_troops,
                "ally_troops": game_state.ally_troops,
                "win_condition": game_state.win_condition
            },
            "action": asdict(action),
            "model_outputs": {
                "action_logits": model_outputs.get('action_logits', []).tolist() if isinstance(model_outputs.get('action_logits'), np.ndarray) else model_outputs.get('action_logits', []),
                "card_logits": model_outputs.get('card_logits', []).tolist() if isinstance(model_outputs.get('card_logits'), np.ndarray) else model_outputs.get('card_logits', []),
                "position": model_outputs.get('position', []).tolist() if isinstance(model_outputs.get('position'), np.ndarray) else model_outputs.get('position', []),
                "confidence": float(model_outputs.get('confidence', 0.0)),
                "value": float(model_outputs.get('value', 0.0))
            },
            "reasoning": action.reasoning
        }
        
        # Write to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(decision_log) + '\n')
        
        # Log to console
        self.logger.info(f"AI Decision: {action.action_type} - {action.reasoning}")
        if action.action_type == "place_card":
            self.logger.info(f"  Card: {action.card_name} (slot {action.card_slot})")
            self.logger.info(f"  Position: ({action.target_x}, {action.target_y}) - Zone: {action.target_zone}")
            self.logger.info(f"  Confidence: {action.confidence:.3f}")
    
    def get_recent_decisions(self, count: int = 10) -> List[Dict]:
        """Get recent decisions from log file"""
        decisions = []
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-count:]:
                    try:
                        decisions.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return decisions

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

class ClashRoyalePPO(nn.Module):
    """PPO-based AI model for Clash Royale decision making"""
    
    def __init__(self, input_size: int = 15, hidden_size: int = 256, num_actions: int = 5):
        super(ClashRoyalePPO, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=2)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Policy heads (output action probabilities)
        self.action_policy = nn.Linear(hidden_size, num_actions)
        self.card_policy = nn.Linear(hidden_size, 4)
        self.position_policy = nn.Linear(hidden_size, 2)  # Mean for continuous position
        self.position_std = nn.Parameter(torch.ones(2) * 0.1)  # Learnable std
        self.zone_policy = nn.Linear(hidden_size, 6)
        
        # Value function
        self.value_head = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last output
        last_output = attn_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # Policy outputs (probabilities)
        action_logits = self.action_policy(last_output)
        card_logits = self.card_policy(last_output)
        position_mean = torch.tanh(self.position_policy(last_output))  # [-1, 1]
        zone_logits = self.zone_policy(last_output)
        
        # Value function
        value = self.value_head(last_output)
        
        return {
            'action_logits': action_logits,
            'card_logits': card_logits,
            'position_mean': position_mean,
            'position_std': self.position_std.expand_as(position_mean),
            'zone_logits': zone_logits,
            'value': value
        }
    
    def get_action_and_value(self, x):
        """Sample actions and return log probabilities and values"""
        outputs = self.forward(x)
        
        # Sample actions
        action_dist = torch.distributions.Categorical(logits=outputs['action_logits'])
        card_dist = torch.distributions.Categorical(logits=outputs['card_logits'])
        zone_dist = torch.distributions.Categorical(logits=outputs['zone_logits'])
        position_dist = torch.distributions.Normal(outputs['position_mean'], outputs['position_std'])
        
        action = action_dist.sample()
        card = card_dist.sample()
        zone = zone_dist.sample()
        position = position_dist.sample()
        
        # Calculate log probabilities
        action_log_prob = action_dist.log_prob(action)
        card_log_prob = card_dist.log_prob(card)
        zone_log_prob = zone_dist.log_prob(zone)
        position_log_prob = position_dist.log_prob(position).sum(dim=-1)
        
        total_log_prob = action_log_prob + card_log_prob + zone_log_prob + position_log_prob
        
        return {
            'action': action,
            'card': card,
            'zone': zone,
            'position': position,
            'log_prob': total_log_prob,
            'value': outputs['value'].squeeze(-1)
        }
    
    def evaluate_actions(self, x, actions):
        """Evaluate given actions and return log probabilities, values, and entropy"""
        outputs = self.forward(x)
        
        action_dist = torch.distributions.Categorical(logits=outputs['action_logits'])
        card_dist = torch.distributions.Categorical(logits=outputs['card_logits'])
        zone_dist = torch.distributions.Categorical(logits=outputs['zone_logits'])
        position_dist = torch.distributions.Normal(outputs['position_mean'], outputs['position_std'])
        
        action_log_prob = action_dist.log_prob(actions['action'])
        card_log_prob = card_dist.log_prob(actions['card'])
        zone_log_prob = zone_dist.log_prob(actions['zone'])
        position_log_prob = position_dist.log_prob(actions['position']).sum(dim=-1)
        
        total_log_prob = action_log_prob + card_log_prob + zone_log_prob + position_log_prob
        
        entropy = (action_dist.entropy() + card_dist.entropy() + 
                  zone_dist.entropy() + position_dist.entropy().sum(dim=-1))
        
        return total_log_prob, outputs['value'].squeeze(-1), entropy

class PPOBuffer:
    """Buffer for PPO training data"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self):
        return (torch.stack(self.states), torch.stack(self.actions), 
                torch.tensor(self.rewards), torch.stack(self.values),
                torch.stack(self.log_probs), torch.tensor(self.dones))
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)

class ClashRoyalePPOAgent:
    """PPO-based agent for Clash Royale"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize PPO model
        self.model = ClashRoyalePPO(input_size=15, hidden_size=256).to(self.device)
        
        # Load pretrained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # PPO optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        
        # PPO buffer
        self.buffer = PPOBuffer()
        
        # Game state tracking
        self.current_state = GameState()
        self.state_history = deque(maxlen=100)
        
        # Decision logging
        self.decision_logger = DecisionLogger()
        
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
        
        # Zone mapping
        self.zone_map = {
            0: "bottom_left",
            1: "bottom_center", 
            2: "bottom_right",
            3: "top_left",
            4: "top_center",
            5: "top_right"
        }
        
        # PPO hyperparameters
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.update_epochs = 4
        self.batch_size = 64
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
    def setup_zmq(self):
        """Setup ZeroMQ connections to receive game data"""
        self.context = zmq.Context()
        
        # Subscribe to all game data
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect("tcp://localhost:5560")  # Elixir
        self.sub_socket.connect("tcp://localhost:5590")  # Cards
        self.sub_socket.connect("tcp://localhost:5580")  # Troops
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
            troops = self._parse_troops(data)
            self.current_state.troops_on_field = troops
            
            # Separate enemy and ally troops
            self.current_state.enemy_troops = [t for t in troops if t.get('team') == 'enemy']
            self.current_state.ally_troops = [t for t in troops if t.get('team') == 'ally']
            
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
        """Convert current game state to enhanced feature vector"""
        # Basic features
        elixir = self.current_state.elixir
        cards_count = len(self.current_state.cards_in_hand)
        troops_count = len(self.current_state.troops_on_field)
        
        # Enhanced features
        enemy_troops = self.current_state.enemy_troops
        ally_troops = self.current_state.ally_troops
        
        # Enemy troop features
        enemy_count = len(enemy_troops)
        enemy_avg_x = np.mean([t.get('x', 0) for t in enemy_troops]) if enemy_troops else 0
        enemy_avg_y = np.mean([t.get('y', 0) for t in enemy_troops]) if enemy_troops else 0
        
        # Ally troop features  
        ally_count = len(ally_troops)
        ally_avg_x = np.mean([t.get('x', 0) for t in ally_troops]) if ally_troops else 0
        ally_avg_y = np.mean([t.get('y', 0) for t in ally_troops]) if ally_troops else 0
        
        # Card features
        card_names = [card.get('name', '') for card in self.current_state.cards_in_hand]
        unique_cards = len(set(card_names))
        
        # Win condition encoding
        win_condition = self.current_state.win_condition
        if win_condition == "win":
            win_encoded = 1.0
        elif win_condition == "lose":
            win_encoded = -1.0
        else:
            win_encoded = 0.0
        
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
        
        features = np.array([
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
            len(self.current_state.cards_in_hand),  # Redundant but kept for compatibility
            time.time() - self.current_state.timestamp  # Time since last update
        ], dtype=np.float32)
        
        return features
    
    def select_action(self, state_features: np.ndarray, training: bool = True) -> Tuple[Action, Dict]:
        """Select action using PPO policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0).unsqueeze(0).to(self.device)
            result = self.model.get_action_and_value(state_tensor)
            
            # Extract sampled actions
            action_idx = result['action'].item()
            card_slot = result['card'].item()
            zone_idx = result['zone'].item()
            position_norm = result['position'].squeeze().cpu().numpy()
            
            action_type = self.action_map[action_idx]
            target_zone = self.zone_map[zone_idx]
            
            # Convert normalized position to game coordinates
            target_x = int((position_norm[0] + 1) * 340 + 200)  # [-1,1] -> [200,880]
            target_y = int((position_norm[1] + 1) * 500 + 200)   # [-1,1] -> [200,1200]
            
            card_name = None
            if action_type == "place_card" and card_slot < len(self.current_state.cards_in_hand):
                card_name = self.current_state.cards_in_hand[card_slot].get('name', 'Unknown')
            
            # Generate reasoning
            confidence = torch.sigmoid(result['log_prob']).item()
            reasoning = self._generate_reasoning(action_type, card_name, target_zone, confidence)
            
            action = Action(
                action_type=action_type,
                card_slot=card_slot if action_type == "place_card" else None,
                target_x=target_x if action_type == "place_card" else None,
                target_y=target_y if action_type == "place_card" else None,
                target_zone=target_zone if action_type == "place_card" else None,
                confidence=confidence,
                reasoning=reasoning,
                card_name=card_name,
                timestamp=time.time()
            )
            
            # Store for training
            if training:
                actions_dict = {
                    'action': result['action'],
                    'card': result['card'],
                    'zone': result['zone'],
                    'position': result['position']
                }
                return action, {
                    'state_tensor': state_tensor,
                    'actions': actions_dict,
                    'log_prob': result['log_prob'],
                    'value': result['value']
                }
            
            return action, {}
    
    def _generate_reasoning(self, action_type: str, card_name: Optional[str], target_zone: Optional[str], confidence: float) -> str:
        """Generate human-readable reasoning for AI decisions"""
        if action_type == "place_card":
            reasoning = f"Playing {card_name} at {target_zone} zone"
            
            # Add context-based reasoning
            if len(self.current_state.enemy_troops) > len(self.current_state.ally_troops):
                reasoning += " to counter enemy advantage"
            elif self.current_state.elixir > 8:
                reasoning += " to use excess elixir"
            elif len(self.current_state.cards_in_hand) == 4:
                reasoning += " to cycle cards"
            else:
                reasoning += " based on current game state"
                
            reasoning += f" (confidence: {confidence:.2f})"
            
        elif action_type == "wait":
            reasoning = f"Waiting for better opportunity (confidence: {confidence:.2f})"
            if self.current_state.elixir < 3:
                reasoning += " - need more elixir"
            elif len(self.current_state.cards_in_hand) < 2:
                reasoning += " - waiting for more cards"
                
        elif action_type == "defend":
            reasoning = f"Defending against enemy troops (confidence: {confidence:.2f})"
            
        else:
            reasoning = f"Executing {action_type} (confidence: {confidence:.2f})"
            
        return reasoning
    
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
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        return advantages, returns
    
    def ppo_update(self):
        """Perform PPO update"""
        if len(self.buffer) < self.batch_size:
            return
        
        # Get data from buffer
        states, actions_dict, rewards, values, old_log_probs, dones = self.buffer.get()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update loop
        for _ in range(self.update_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = {k: v[batch_indices] for k, v in actions_dict.items()}
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Evaluate current policy
                new_log_probs, new_values, entropy = self.model.evaluate_actions(batch_states, batch_actions)
                
                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(new_values, batch_returns)
                entropy_loss = -entropy.mean()
                
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Clear buffer
        self.buffer.clear()
    
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
                    action, model_outputs = self.select_action(state_features, training)
                    
                    # Log decision
                    self.decision_logger.log_decision(self.current_state, action, model_outputs)
                    
                    # Execute action
                    self.execute_action(action)
                    
                    # Store experience for PPO training
                    if training and model_outputs:
                        reward = self._calculate_reward()
                        done = self.current_state.win_condition in ["win", "lose"]
                        
                        self.buffer.store(
                            model_outputs['state_tensor'].squeeze(0),
                            model_outputs['actions'],
                            reward,
                            model_outputs['value'],
                            model_outputs['log_prob'],
                            done
                        )
                        
                        # Update model when buffer is full or episode ends
                        if len(self.buffer) >= self.batch_size or done:
                            self.ppo_update()
                    
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
        }, path)
        print(f"PPO Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"PPO Model loaded from {path}")
    
    def cleanup(self):
        """Clean up resources"""
        self.sub_socket.close()
        self.pub_socket.close()
        self.context.term()

def train_ppo_model(data_file: str, epochs: int = 100):
    """Train PPO model using historical data (behavioral cloning pre-training)"""
    print(f"Pre-training PPO model with data from {data_file}")
    
    # Create dataset
    dataset = ClashRoyaleDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = ClashRoyalePPO().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Pre-training loop (behavioral cloning)
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (features, rewards) in enumerate(dataloader):
            optimizer.zero_grad()
            
            outputs = model(features)
            # Simple value prediction loss for pre-training
            value_loss = nn.MSELoss()(outputs['value'].squeeze(), rewards)
            
            value_loss.backward()
            optimizer.step()
            
            total_loss += value_loss.item()
        
        print(f"Pre-training Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model

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
        train_ppo_model(args.data_file, args.epochs)
    else:
        agent = ClashRoyalePPOAgent(args.model_path)
        agent.run(training=True)
