"""
Clash Royale PPO AI Agent

Simple PPO-based AI for Clash Royale gameplay that makes in-game decisions only.
"""

import os
import sys
import time
import json
import threading
import random
import numpy as np
import cv2
import zmq
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import logging
from datetime import datetime

# Add parent directory to path to import gameplayer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gameplayer'))
from gameplayer import GamePlayer

load_dotenv()


# Load zone coordinates from locations.txt if available. Falls back to None.
def _load_zone_coords():
    """Load zone coordinates.

    Priority:
      1. Load from coordinates.json if present.
      2. Parse locations.txt and write coordinates.json for future runs.
      3. Fall back to built-in defaults and write coordinates.json.
    Returns a dict mapping zone_name (str) -> (x, y).
    """
    base = os.path.dirname(__file__)
    json_path = os.path.join(base, 'coordinates.json')
    loc_path = os.path.join(base, 'locations.txt')

    # Try to load explicit JSON first
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as jf:
                raw = json.load(jf)
            coords = {}
            for k, v in raw.items():
                try:
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        coords[k] = (int(v[0]), int(v[1]))
                except Exception:
                    continue
            if coords:
                return coords
    except Exception:
        # ignore JSON errors and fall through to parsing locations.txt or defaults
        pass

    coords = {}
    # If locations.txt exists, parse it and create coordinates.json
    if os.path.exists(loc_path):
        try:
            with open(loc_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        parts = line.split(':', 1)
                        idx = int(parts[0].strip())
                        # Map index to zone name
                        zone_names = ["bottom_left", "bottom_center", "bottom_right", "top_left", "top_center", "top_right"]
                        zone_name = zone_names[idx] if 0 <= idx < len(zone_names) else None
                        if not zone_name:
                            continue
                        lpar = line.rfind('(')
                        rpar = line.rfind(')')
                        if lpar != -1 and rpar != -1 and rpar > lpar:
                            xy = line[lpar+1:rpar]
                            x_str, y_str = xy.split(',', 1)
                            x = int(x_str.strip())
                            y = int(y_str.strip())
                            coords[zone_name] = (x, y)
                    except Exception:
                        continue
        except Exception:
            coords = {}

    # If still empty, use sensible defaults (from provided values)
    if not coords:
        coords = {
            # Main placement zones (for PPO model)
            "bottom_left": (1390, 762),
            "bottom_center": (1611, 658),
            "bottom_right": (1816, 725),
            "top_left": (1404, 472),
            "top_center": (1606, 489),
            "top_right": (1810, 489),
            
            # Defensive zones
            "defend_left": (1459, 546),
            "defend_right": (1752, 546),
            "defend_center_top": (1606, 491),
            "defend_center": (1602, 660),
            
            # Tactical zones
            "back_center": (1611, 850),           # Back support position
            "front_center": (1611, 580),          # Front push position
            "counter_left": (1450, 650),          # Counter-push left
            "counter_right": (1750, 650),         # Counter-push right
            "split_left": (1400, 720),            # Split push left
            "split_right": (1800, 720),           # Split push right
            "tank_center": (1611, 600),           # Tank push center
            "support_back": (1611, 800),          # Back line support
            "air_support_center": (1611, 650),    # Air unit support
            "default_placement": (1611, 680)      # Default fallback
        }

    # Persist coordinates to coordinates.json for future runs
    try:
        to_save = {k: [v[0], v[1]] for k, v in coords.items()}
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(to_save, jf, indent=2)
    except Exception:
        pass

    return coords

# zone_coords is a mapping from zone index -> (x,y) in screen coordinates
ZONE_COORDS = _load_zone_coords()

@dataclass
class GameState:
    """Current state of the game"""
    elixir: int = 0
    cards_in_hand: List[Dict] = None
    troops_on_field: List[Dict] = None
    win_condition: str = "ongoing"
    timestamp: float = 0.0
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
    """Action the AI can take"""
    action_type: str  # "place_card", "wait", "defend"
    card_slot: Optional[int] = None
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    target_zone: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    card_name: Optional[str] = None
    timestamp: float = 0.0

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

class ClashRoyalePPO(nn.Module):
    """PPO model for Clash Royale"""
    
    def __init__(self, input_size: int = 15, hidden_size: int = 256, num_actions: int = 3):
        super(ClashRoyalePPO, self).__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # LSTM for sequences
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=2)
        
        # Attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Policy heads
        self.action_policy = nn.Linear(hidden_size, num_actions)
        self.card_policy = nn.Linear(hidden_size, 4)
        self.position_policy = nn.Linear(hidden_size, 2)
        self.position_std = nn.Parameter(torch.ones(2) * 0.1)
        self.zone_policy = nn.Linear(hidden_size, 16)
        
        # Value function
        self.value_head = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        lstm_out, _ = self.lstm(features)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_output = self.dropout(attn_out[:, -1, :])
        
        return {
            'action_logits': self.action_policy(last_output),
            'card_logits': self.card_policy(last_output),
            'position_mean': torch.tanh(self.position_policy(last_output)),
            'position_std': self.position_std.expand_as(torch.tanh(self.position_policy(last_output))),
            'zone_logits': self.zone_policy(last_output),
            'value': self.value_head(last_output)
        }
    
    def get_action_and_value(self, x):
        """Sample actions and return log probabilities and values"""
        outputs = self.forward(x)
        
        action_dist = torch.distributions.Categorical(logits=outputs['action_logits'])
        card_dist = torch.distributions.Categorical(logits=outputs['card_logits'])
        zone_dist = torch.distributions.Categorical(logits=outputs['zone_logits'])
        position_dist = torch.distributions.Normal(outputs['position_mean'], outputs['position_std'])
        
        action = action_dist.sample()
        card = card_dist.sample()
        zone = zone_dist.sample()
        position = position_dist.sample()
        
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
        """Evaluate actions for PPO update"""
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

class ClashRoyalePPOAgent:
    """PPO agent for Clash Royale"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize PPO model (3 actions only)
        self.model = ClashRoyalePPO(input_size=15, hidden_size=256, num_actions=3).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.buffer = PPOBuffer()
        
        # Game state
        self.current_state = GameState()
        
        # ZeroMQ setup
        self.setup_zmq()
        
        # Game player
        self.game_player = GamePlayer()
        
        # Action mapping (in-game only)
        self.action_map = {
            0: "wait",
            1: "place_card",
            2: "defend"
        }
        
        # Zone mapping - list of zone names for indexing from model output
        self.zone_names = [
            "bottom_left", "bottom_center", "bottom_right",
            "top_left", "top_center", "top_right",
            "defend_left", "defend_right", "defend_center", "defend_center_top",
            "back_center", "front_center", "counter_left", "counter_right",
            "split_left", "tank_center"
        ]
        
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
        """Setup ZeroMQ connections"""
        self.context = zmq.Context()
        
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect("tcp://localhost:5560")  # Elixir
        self.sub_socket.connect("tcp://localhost:5590")  # Cards
        self.sub_socket.connect("tcp://localhost:5580")  # Troops
        self.sub_socket.connect("tcp://localhost:5570")  # Win detection
        
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"ecount|")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"cards|")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"troops|")
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"winner|")
        
    def update_game_state(self, topic: str, data: str):
        """Update game state from ZeroMQ data"""
        if topic == "ecount":
            self.current_state.elixir = int(data)
        elif topic == "cards":
            self.current_state.cards_in_hand = self._parse_cards(data)
        elif topic == "troops":
            troops = self._parse_troops(data)
            self.current_state.troops_on_field = troops
            self.current_state.enemy_troops = [t for t in troops if t.get('team') == 'enemy']
            self.current_state.ally_troops = [t for t in troops if t.get('team') == 'ally']
        elif topic == "winner":
            self.current_state.win_condition = data
        
        self.current_state.timestamp = time.time()
        
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
        """Parse troop data"""
        try:
            troops_json = json.loads(troops_data)
            troops = []
            if isinstance(troops_json, list) and troops_json:
                for result in troops_json:
                    predictions = result.get("predictions", {}).get("predictions", [])
                    for pred in predictions:
                        class_name = pred.get("class", "").lower()
                        team = "enemy" if "enemy" in class_name else "ally"
                        
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
    
    def get_state_features(self) -> np.ndarray:
        """Convert game state to feature vector"""
        elixir = self.current_state.elixir
        cards_count = len(self.current_state.cards_in_hand)
        troops_count = len(self.current_state.troops_on_field)
        
        enemy_troops = self.current_state.enemy_troops
        ally_troops = self.current_state.ally_troops
        
        enemy_count = len(enemy_troops)
        enemy_avg_x = np.mean([t.get('x', 0) for t in enemy_troops]) if enemy_troops else 0
        enemy_avg_y = np.mean([t.get('y', 0) for t in enemy_troops]) if enemy_troops else 0
        
        ally_count = len(ally_troops)
        ally_avg_x = np.mean([t.get('x', 0) for t in ally_troops]) if ally_troops else 0
        ally_avg_y = np.mean([t.get('y', 0) for t in ally_troops]) if ally_troops else 0
        
        card_names = [card.get('name', '') for card in self.current_state.cards_in_hand]
        unique_cards = len(set(card_names))
        
        win_condition = self.current_state.win_condition
        win_encoded = 1.0 if win_condition == "win" else (-1.0 if win_condition == "lose" else 0.0)
        
        troop_balance = ally_count - enemy_count
        
        if enemy_troops and ally_troops:
            min_distance = min([
                np.sqrt((e.get('x', 0) - a.get('x', 0))**2 + (e.get('y', 0) - a.get('y', 0))**2)
                for e in enemy_troops for a in ally_troops
            ])
        else:
            min_distance = 1000.0
        
        features = np.array([
            elixir, cards_count, troops_count, enemy_count, ally_count,
            enemy_avg_x, enemy_avg_y, ally_avg_x, ally_avg_y, unique_cards,
            win_encoded, troop_balance, min_distance, len(self.current_state.cards_in_hand),
            time.time() - self.current_state.timestamp
        ], dtype=np.float32)
        
        return features
    
    def select_action(self, state_features: np.ndarray, training: bool = True) -> Tuple[Action, Dict]:
        """Select action using PPO policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0).unsqueeze(0).to(self.device)
            result = self.model.get_action_and_value(state_tensor)
            
            action_idx = result['action'].item()
            card_slot = result['card'].item()
            zone_idx = result['zone'].item()
            position_norm = result['position'].squeeze().cpu().numpy()
            
            action_type = self.action_map[action_idx]
            target_zone = self.zone_names[zone_idx]
            # Prefer using zone coordinates if available (user-supplied locations.txt)
            target_x = None
            target_y = None
            if action_type == "place_card":
                if ZONE_COORDS and target_zone in ZONE_COORDS:
                    # Use the canonical coordinate for the inferred zone
                    tx, ty = ZONE_COORDS[target_zone]
                    target_x = int(tx)
                    target_y = int(ty)
                else:
                    # Fallback to the sampled continuous position mapping
                    target_x = int((position_norm[0] + 1) * 340 + 200)
                    target_y = int((position_norm[1] + 1) * 500 + 200)
            
            card_name = None
            if action_type == "place_card" and card_slot < len(self.current_state.cards_in_hand):
                card_name = self.current_state.cards_in_hand[card_slot].get('name', 'Unknown')
            
            confidence = torch.sigmoid(result['log_prob']).item()
            reasoning = f"{action_type}"
            if action_type == "place_card":
                reasoning += f" {card_name} at {target_zone}"
            
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
    
    def execute_action(self, action: Action):
        """Execute action using GamePlayer"""
        try:
            if action.action_type == "place_card" and action.card_slot is not None:
                if action.target_x is not None and action.target_y is not None:
                    # Special spell targeting
                    if self._is_spell(action.card_name):
                        spell_pos = self._get_spell_target(action.card_name)
                        if spell_pos:
                            self.game_player.place_card(action.card_slot, spell_pos['x'], spell_pos['y'])
                            print(f"AI cast {action.card_name} at {spell_pos['target']} ({spell_pos['x']}, {spell_pos['y']})")
                        else:
                            print(f"AI held {action.card_name} - no good targets")
                    else:
                        # Use model's zone selection
                        self.game_player.place_card(action.card_slot, action.target_x, action.target_y)
                        print(f"AI placed {action.card_name} at {action.target_zone} ({action.target_x}, {action.target_y})")
            elif action.action_type == "defend":
                # Use model's zone for defense
                defensive_card = self._select_defensive_card()
                if defensive_card and action.target_x and action.target_y:
                    self.game_player.place_card(defensive_card['slot'], action.target_x, action.target_y)
                    print(f"AI defending with {defensive_card['name']} at {action.target_zone}")
                else:
                    print("AI defending but no suitable defensive cards available")
            elif action.action_type == "wait":
                time.sleep(0.5)
                print("AI waiting...")
        except Exception as e:
            print(f"Error executing action: {e}")
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            next_value = 0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        return advantages, returns
    
    def ppo_update(self):
        """Perform PPO update"""
        if len(self.buffer) < self.batch_size:
            return
        
        states, actions_dict, rewards, values, old_log_probs, dones = self.buffer.get()
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.update_epochs):
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = {k: v[batch_indices] for k, v in actions_dict.items()}
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                new_log_probs, new_values, entropy = self.model.evaluate_actions(batch_states, batch_actions)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(new_values, batch_returns)
                entropy_loss = -entropy.mean()
                
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        self.buffer.clear()
    
    def run(self, training: bool = True):
        """Main game loop"""
        print("Starting Clash Royale PPO Agent...")
        
        last_action_time = 0
        action_interval = 2.0
        
        try:
            while True:
                try:
                    msg = self.sub_socket.recv(zmq.NOBLOCK)
                    topic, data = msg.decode().split("|", 1)
                    self.update_game_state(topic, data)
                except zmq.Again:
                    pass
                
                current_time = time.time()
                
                if current_time - last_action_time >= action_interval:
                    state_features = self.get_state_features()
                    action, model_outputs = self.select_action(state_features, training)
                    
                    print(f"AI Decision: {action.reasoning}")
                    self.execute_action(action)
                    
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
                        
                        if len(self.buffer) >= self.batch_size or done:
                            self.ppo_update()
                    
                    last_action_time = current_time
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Shutting down AI agent...")
        finally:
            self.cleanup()
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on game state"""
        reward = 0.0
        reward += self.current_state.elixir * 0.1
        reward += len(self.current_state.cards_in_hand) * 0.05
        
        if self.current_state.win_condition == "win":
            reward += 10.0
        elif self.current_state.win_condition == "lose":
            reward -= 10.0
        
        return reward
    
    def save_model(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"PPO Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"PPO Model loaded from {path}")
    
    def _select_defensive_card(self) -> Optional[Dict]:
        """Select best defensive card from hand"""
        if not self.current_state.cards_in_hand:
            return None
        
        # Deck-specific defensive priorities
        deck_defensive_cards = {
            'knight': 9,      # Best tank for defense
            'musketeer': 7,   # Good range defense
            'archers': 6,     # Cheap air defense
            'minions': 5,     # Air defense/distraction
            'mini pekka': 8   # High damage defense
        }
        
        best_card = None
        best_score = 0
        
        enemy_count = len(self.current_state.enemy_troops)
        air_enemies = sum(1 for t in self.current_state.enemy_troops 
                         if 'dragon' in t.get('type', '').lower() or 'balloon' in t.get('type', '').lower() or 'minion' in t.get('type', '').lower() or 'barrel' in t.get('type', '').lower())
        
        for card in self.current_state.cards_in_hand:
            card_name = card.get('name', '').lower()
            
            for def_name, base_score in deck_defensive_cards.items():
                if def_name in card_name:
                    score = base_score
                    
                    # Boost for air defense against air units
                    if air_enemies > 0 and def_name in ['archers', 'musketeer', 'minions']:
                        score += 3
                    
                    # Boost tank cards when many enemies
                    if enemy_count > 2 and def_name in ['knight', 'mini pekka']:
                        score += 2
                    
                    if score > best_score:
                        best_score = score
                        best_card = card
                    break
        
        # If no defensive cards, use cheapest card available
        if not best_card and self.current_state.cards_in_hand:
            best_card = self.current_state.cards_in_hand[0]  # Fallback
        
        return best_card or (self.current_state.cards_in_hand[0] if self.current_state.cards_in_hand else None)
    
    def _get_defensive_position(self) -> Dict:
        """Get tactical defensive position based on enemy threats"""
        # Define defensive zones in priority order
        defensive_zones = ["defend_left", "defend_right", "defend_center_top", "defend_center"]
        
        # Filter to only zones that exist in ZONE_COORDS
        available_zones = [zone for zone in defensive_zones if zone in ZONE_COORDS]
        
        if not available_zones:
            # Fallback to defend_center coordinate if no defensive zones defined
            return {'x': 1602, 'y': 660, 'zone': 'defend_center'}
        
        # Randomly select a defensive zone
        selected_zone = random.choice(available_zones)
        x, y = ZONE_COORDS[selected_zone]
        
        return {'x': x, 'y': y, 'zone': selected_zone}
    
    def _is_spell(self, card_name: str) -> bool:
        """Check if card is a spell"""
        if not card_name:
            return False
        name = card_name.lower()
        return 'fireball' in name or 'arrows' in name
    
    def _get_spell_target(self, card_name: str) -> Optional[Dict]:
        """Get optimal spell target position"""
        if not card_name:
            return None
        
        name = card_name.lower()
        enemy_troops = self.current_state.enemy_troops
        
        if 'fireball' in name:
            # Fireball: Target clusters of enemies or high-value targets
            if len(enemy_troops) >= 2:
                # Find cluster center - use actual enemy positions
                avg_x = sum(t.get('x', 540) for t in enemy_troops) / len(enemy_troops)
                avg_y = sum(t.get('y', 400) for t in enemy_troops) / len(enemy_troops)
                return {'x': int(avg_x), 'y': int(avg_y), 'target': 'enemy_cluster'}
            elif len(enemy_troops) == 1:
                # Single high-value target - use actual enemy position
                target = enemy_troops[0]
                return {'x': int(target.get('x', 540)), 'y': int(target.get('y', 400)), 'target': 'single_enemy'}
            else:
                # No enemies, use defend_center_top as default spell location
                if 'defend_center_top' in ZONE_COORDS:
                    x, y = ZONE_COORDS['defend_center_top']
                    return {'x': x, 'y': y, 'target': 'defend_center_top'}
        
        elif 'arrows' in name:
            # Arrows: Target swarms or air units
            air_units = [t for t in enemy_troops if 'minion' in t.get('type', '').lower() or 'dragon' in t.get('type', '').lower()]
            if air_units:
                # Use actual air unit positions
                avg_x = sum(t.get('x', 540) for t in air_units) / len(air_units)
                avg_y = sum(t.get('y', 400) for t in air_units) / len(air_units)
                return {'x': int(avg_x), 'y': int(avg_y), 'target': 'air_swarm'}
            elif len(enemy_troops) >= 3:
                # Target swarm - use actual enemy positions
                avg_x = sum(t.get('x', 540) for t in enemy_troops) / len(enemy_troops)
                avg_y = sum(t.get('y', 400) for t in enemy_troops) / len(enemy_troops)
                return {'x': int(avg_x), 'y': int(avg_y), 'target': 'ground_swarm'}
            else:
                # No good targets, use defend_center as default
                if 'defend_center' in ZONE_COORDS:
                    x, y = ZONE_COORDS['defend_center']
                    return {'x': x, 'y': y, 'target': 'defend_center'}
        
        return None  # No good target found
    
    def _get_tactical_position(self, card_name: str) -> Dict:
        """Get tactical position for troop cards using ZONE_COORDS"""
        if not card_name:
            # Use default_placement region
            if 'default_placement' in ZONE_COORDS:
                x, y = ZONE_COORDS['default_placement']
                return {'x': x, 'y': y, 'strategy': 'default'}
            return {'x': 1611, 'y': 680, 'strategy': 'default'}
        
        name = card_name.lower()
        enemy_troops = self.current_state.enemy_troops
        ally_troops = self.current_state.ally_troops
        
        # Giant: Front tank position
        if 'giant' in name:
            if enemy_troops:
                # Push toward enemies - use actual enemy position for targeting
                enemy_avg_x = sum(t.get('x', 540) for t in enemy_troops) / len(enemy_troops)
                # But use tank_center y-coordinate for consistency
                y = ZONE_COORDS.get('tank_center', (1611, 600))[1]
                return {'x': int(enemy_avg_x), 'y': y, 'strategy': 'tank_push'}
            # Use tank_center region
            if 'tank_center' in ZONE_COORDS:
                x, y = ZONE_COORDS['tank_center']
                return {'x': x, 'y': y, 'strategy': 'tank_center'}
            return {'x': 1611, 'y': 600, 'strategy': 'center_push'}
        
        # Knight: Counter-push or defense
        elif 'knight' in name:
            if len(enemy_troops) > len(ally_troops):
                # Defensive position - use defend_center
                if 'defend_center' in ZONE_COORDS:
                    x, y = ZONE_COORDS['defend_center']
                    return {'x': x, 'y': y, 'strategy': 'defensive_tank'}
                return {'x': 1602, 'y': 660, 'strategy': 'defensive_tank'}
            else:
                # Counter-push left
                if 'counter_left' in ZONE_COORDS:
                    x, y = ZONE_COORDS['counter_left']
                    return {'x': x, 'y': y, 'strategy': 'counter_push'}
                return {'x': 1450, 'y': 650, 'strategy': 'counter_push'}
        
        # Mini PEKKA: High damage counter
        elif 'mini pekka' in name:
            if enemy_troops:
                # Target closest enemy - use actual enemy position
                closest = min(enemy_troops, key=lambda t: t.get('y', 0))
                return {'x': int(closest.get('x', 1611)), 'y': int(closest.get('y', 650)), 'strategy': 'counter_attack'}
            # Use front_center for push support
            if 'front_center' in ZONE_COORDS:
                x, y = ZONE_COORDS['front_center']
                return {'x': x, 'y': y, 'strategy': 'push_support'}
            return {'x': 1611, 'y': 580, 'strategy': 'push_support'}
        
        # Musketeer: Back support
        elif 'musketeer' in name:
            if 'support_back' in ZONE_COORDS:
                x, y = ZONE_COORDS['support_back']
                return {'x': x, 'y': y, 'strategy': 'back_support'}
            return {'x': 1611, 'y': 800, 'strategy': 'back_support'}
        
        # Archers: Split or support
        elif 'archers' in name:
            if len(ally_troops) > 0:
                # Support existing push - use ally position x with support y
                ally_avg_x = sum(t.get('x', 1611) for t in ally_troops) / len(ally_troops)
                y = ZONE_COORDS.get('back_center', (1611, 850))[1]
                return {'x': int(ally_avg_x), 'y': y, 'strategy': 'support_push'}
            # Use split_left region
            if 'split_left' in ZONE_COORDS:
                x, y = ZONE_COORDS['split_left']
                return {'x': x, 'y': y, 'strategy': 'split_push'}
            return {'x': 1400, 'y': 720, 'strategy': 'split_push'}
        
        # Minions: Air support
        elif 'minions' in name:
            if ally_troops:
                # Support ground troops - use ally position x with air support y
                ally_avg_x = sum(t.get('x', 1611) for t in ally_troops) / len(ally_troops)
                y = ZONE_COORDS.get('air_support_center', (1611, 650))[1]
                return {'x': int(ally_avg_x), 'y': y, 'strategy': 'air_support'}
            # Use air_support_center region
            if 'air_support_center' in ZONE_COORDS:
                x, y = ZONE_COORDS['air_support_center']
                return {'x': x, 'y': y, 'strategy': 'air_push'}
            return {'x': 1611, 'y': 650, 'strategy': 'air_push'}
        
        # Default position - use default_placement region
        if 'default_placement' in ZONE_COORDS:
            x, y = ZONE_COORDS['default_placement']
            return {'x': x, 'y': y, 'strategy': 'default'}
        return {'x': 1611, 'y': 680, 'strategy': 'default'}
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'sub_socket'):
            self.sub_socket.close()
        if hasattr(self, 'context'):
            self.context.term()

def train_ppo_model(data_file: str, epochs: int = 100):
    """Simple training function"""
    print(f"Training PPO model with {epochs} epochs")
    # Simplified training - just create and return model
    model = ClashRoyalePPO()
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clash Royale PPO AI Agent")
    parser.add_argument("--mode", choices=["train", "play"], default="play")
    parser.add_argument("--model-path", type=str, help="Path to saved model")
    parser.add_argument("--epochs", type=int, default=100)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_ppo_model("dummy_data.jsonl", args.epochs)
    else:
        agent = ClashRoyalePPOAgent(args.model_path)
        agent.run(training=True)