# Clash Royale Deep Learning AI

A clean, simplified PPO-based AI for Clash Royale that makes **in-game decisions only**. Perfect for beginners to deep learning and reinforcement learning.

## 🎯 What This AI Does

**In-Game Actions Only:**
- ✅ **Wait** - Patience for better opportunities
- ✅ **Place Card** - Strategic card placement with position/zone selection
- ✅ **Defend** - Defensive positioning and strategy

**What It Doesn't Do:**
- ❌ Start/stop matches (you control this manually)
- ❌ Navigate menus or UI elements
- ❌ Handle non-gameplay interactions

## 🧠 Deep Learning Explained (For Beginners)

### What is PPO (Proximal Policy Optimization)?
Think of PPO as teaching the AI like training a pet:

1. **Policy** = The AI's "brain" that decides what action to take
2. **Reward** = Points for good moves (winning, smart plays) and penalties for bad ones
3. **Learning** = Gradually improving decisions based on rewards

### Why PPO vs Other Methods?
- **Stable Learning**: Won't "forget" good strategies suddenly
- **Sample Efficient**: Learns faster from fewer games
- **Continuous Actions**: Naturally handles precise card placement coordinates

### Neural Network Architecture
```
Game State (15 numbers) → Neural Network → Action Decision
     ↓                         ↓              ↓
[elixir: 7,              [Hidden Layers]   [place_card,
 cards: 4,                with LSTM &       slot: 2,
 enemies: 2, ...]         Attention]        x: 450, y: 600]
```

## 🚀 Quick Start Guide

### Step 1: Install Dependencies
```bash
cd deeplearning
pip install torch torchvision numpy opencv-python pyzmq python-dotenv
```

### Step 2: Test the AI (No Modules Required)
```bash
# Test the AI with simulated game data
python test_ai_standalone.py
```
This runs a complete simulation showing how the AI learns!

### Step 3: Test Individual Components
```bash
# Test the neural network
python test_ppo_model.py

# Test system integration (requires modules)
python new_integration_layer.py --test-all
```

### Step 4: Run with Real Game (Requires Modules)
```bash
# Start the AI (assumes modules are running)
python main.py --mode play --no-modules
```

## 📚 Deep Learning Tutorial

### Understanding the Code Structure

#### 1. Game State Representation
```python
@dataclass
class GameState:
    elixir: int = 0                    # Current elixir (0-10)
    cards_in_hand: List[Dict] = []     # Available cards
    enemy_troops: List[Dict] = []      # Enemy units on field
    ally_troops: List[Dict] = []       # Your units on field
    win_condition: str = "ongoing"     # Game status
```

#### 2. Neural Network (ClashRoyalePPO)
```python
class ClashRoyalePPO(nn.Module):
    def __init__(self):
        # Feature Extractor: Converts game state to internal representation
        self.feature_extractor = nn.Sequential(...)
        
        # LSTM: Remembers patterns over time
        self.lstm = nn.LSTM(...)
        
        # Attention: Focuses on important game elements
        self.attention = nn.MultiheadAttention(...)
        
        # Policy Heads: Different types of decisions
        self.action_policy = nn.Linear(...)  # What type of action?
        self.card_policy = nn.Linear(...)    # Which card to play?
        self.position_policy = nn.Linear(...) # Where to place it?
        self.value_head = nn.Linear(...)     # How good is this state?
```

#### 3. Learning Process (PPO Algorithm)
```python
def ppo_update(self):
    # 1. Collect experiences from recent games
    states, actions, rewards = self.buffer.get()
    
    # 2. Calculate advantages (how much better/worse than expected)
    advantages = self.compute_gae(rewards, values)
    
    # 3. Update policy (but not too much at once - "clipping")
    ratio = new_probability / old_probability
    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)  # Prevent big changes
    
    # 4. Learn from the experience
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    loss.backward()
    optimizer.step()
```

### Key Concepts Explained

#### Feature Engineering (Converting Game to Numbers)
The AI converts complex game state into 15 numbers:
```python
features = [
    elixir_count,           # Resource management
    cards_available,        # Hand size
    enemy_troop_count,      # Threat assessment
    ally_troop_count,       # Your army strength
    enemy_avg_position_x,   # Where enemies are
    enemy_avg_position_y,   # Spatial awareness
    troop_balance,          # Who's winning the field
    min_distance,           # How close is combat
    # ... and more strategic features
]
```

#### Reward Function (Teaching Right from Wrong)
```python
def calculate_reward(self):
    reward = 0
    reward += elixir * 0.1              # Reward elixir management
    reward += cards_available * 0.05     # Reward having options
    reward += 10.0 if win else -10.0    # Big reward for winning
    reward += troop_balance * 1.0       # Reward field control
    return reward
```

#### Action Selection (How AI Decides)
```python
def select_action(self, game_state):
    # Convert game state to neural network input
    features = self.get_state_features(game_state)
    
    # Neural network predicts action probabilities
    outputs = self.model(features)
    
    # Sample action based on probabilities (exploration vs exploitation)
    action = torch.distributions.Categorical(outputs['action_logits']).sample()
    
    return action
```

## 🎮 Training Modes Explained

### Mode 1: Standalone Testing (No Game Required)
```bash
python test_ai_standalone.py
```
- **What it does**: Simulates Clash Royale games in code
- **Good for**: Understanding how the AI works, debugging, learning
- **Output**: Shows AI decisions and learning progress

### Mode 2: Live Game Training (Requires Modules)
```bash
python main.py --mode play --no-modules
```
- **What it does**: Connects to real game via computer vision modules
- **Good for**: Actual gameplay and real-world training
- **Requirements**: All detection modules must be running

### Mode 3: Data Collection (Passive Learning)
```bash
python main.py --mode collect-data --required-only
```
- **What it does**: Records game data without making decisions
- **Good for**: Gathering training data from human gameplay
- **Output**: Creates dataset for supervised learning

## 🔧 Customization Guide

### Adjusting AI Behavior

#### Make AI More Aggressive
```python
# In _calculate_reward() function
reward += len(ally_troops) * 2.0  # Reward having more troops
reward -= enemy_count * 1.5       # Penalty for letting enemies build up
```

#### Make AI More Defensive
```python
# In _calculate_reward() function
reward += elixir * 0.2            # Higher reward for saving elixir
reward += (10 - elixir) * -0.1    # Penalty for spending too much
```

#### Change Learning Speed
```python
# In ClashRoyalePPOAgent.__init__()
self.clip_ratio = 0.1      # Smaller = more conservative learning
self.learning_rate = 1e-4  # Smaller = slower but more stable learning
self.batch_size = 32       # Smaller = more frequent updates
```

### Adding New Features
```python
# In get_state_features() function
def get_state_features(self):
    # Add your custom features
    tower_health = self.get_tower_health()  # If you can detect this
    elixir_advantage = self.current_state.elixir - estimated_enemy_elixir
    
    features = np.array([
        # ... existing features ...
        tower_health,
        elixir_advantage,
        # Add more features here
    ])
    return features
```

## 📊 Monitoring and Debugging

### Understanding AI Decisions
```bash
# Watch AI reasoning in real-time
python main.py --mode play --no-modules
# Look for output like:
# "AI Decision: place_card Knight at bottom_center (confidence: 0.85)"
```

### Performance Metrics
- **Confidence**: How sure the AI is about its decisions (0.0-1.0)
- **Reward**: Points earned per action (higher = better)
- **Win Rate**: Percentage of games won (track over time)

### Common Issues and Solutions

#### AI Makes Random Decisions
- **Cause**: Not enough training or poor reward function
- **Solution**: Train longer or adjust rewards in `_calculate_reward()`

#### AI Always Does Same Action
- **Cause**: Entropy too low (not exploring enough)
- **Solution**: Increase `entropy_coef` in hyperparameters

#### AI Doesn't Learn
- **Cause**: Learning rate too high/low or bad features
- **Solution**: Adjust `learning_rate` or improve `get_state_features()`

## 🎯 Project Structure (Simplified)

```
deeplearning/
├── clash_royale_ai.py          # 🧠 Main AI brain (PPO model + agent)
├── test_ai_standalone.py       # 🎮 Standalone testing (no modules needed)
├── test_ppo_model.py          # 🔧 Neural network testing
├── main.py                    # 🚀 Entry point for real gameplay
├── new_integration_layer.py   # 🔗 Module integration testing
└── README.md                  # 📖 This guide
```

## 🎓 Learning Path for Beginners

### Week 1: Understanding
1. Run `python test_ai_standalone.py` and watch the AI play
2. Read through `clash_royale_ai.py` comments
3. Experiment with reward function changes

### Week 2: Customization
1. Modify the reward function to change AI behavior
2. Add new features to the state representation
3. Adjust hyperparameters and observe changes

### Week 3: Integration
1. Set up the full module pipeline
2. Run the AI with real game data
3. Monitor performance and iterate

### Week 4: Advanced
1. Implement new action types
2. Add more sophisticated reward shaping
3. Experiment with different neural network architectures

## 🤝 Contributing

**Beginner-Friendly Contributions:**
- Improve reward function for better gameplay
- Add new input features (tower health, cycle tracking)
- Create better visualization of AI decisions
- Write more comprehensive tests

**Advanced Contributions:**
- Implement curriculum learning
- Add multi-agent self-play
- Optimize neural network architecture
- Create web dashboard for monitoring

## 📝 License

This project follows the main repository license. Use responsibly and in accordance with Clash Royale's Terms of Service.

---

**🎉 Ready to start?** Run `python test_ai_standalone.py` and watch your AI learn to play Clash Royale!