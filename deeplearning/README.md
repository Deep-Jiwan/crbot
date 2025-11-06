# Clash Royale Deep Learning AI

A sophisticated PPO-based (Proximal Policy Optimization) AI system for Clash Royale gameplay that integrates computer vision, game state analysis, and reinforcement learning.

## Architecture Overview

### PPO-Based Model
- **Policy Network**: Outputs action probabilities for intelligent decision making
- **Value Network**: Estimates state values for advantage calculation
- **Multi-head Output**: Simultaneous prediction of action, card, position, and zone
- **LSTM + Attention**: Processes temporal game sequences with context awareness

### Key Improvements over DQN
- **More Stable Training**: PPO's clipped objective prevents large policy updates
- **Better Sample Efficiency**: No experience replay needed, learns from recent experiences
- **Continuous Actions**: Handles card placement coordinates naturally
- **Policy Gradient**: Direct optimization of the policy for better convergence

## Quick Start

### 1. Install Dependencies
```bash
cd deeplearning
pip install -r requirements.txt
```

### 2. Test System Components
```bash
# Test all components
python new_integration_layer.py --test-all

# Test individual components
python new_integration_layer.py --test-health
python new_integration_layer.py --test-aggregator
python new_integration_layer.py --test-player
```

### 3. Run AI Modes

#### Data Collection Mode
```bash
# Collect training data (run modules only)
python main.py --mode collect-data --required-only
```

#### Training Mode
```bash
# Pre-train with behavioral cloning + PPO fine-tuning
python main.py --mode train --data-file ../masterreceiver/game_data_log.jsonl --epochs 100
```

#### Play Mode
```bash
# Run AI with trained model
python main.py --mode play --model-path models/ppo_model.pth

# Run without starting modules (if already running)
python main.py --mode play --model-path models/ppo_model.pth --no-modules
```

#### Evaluation Mode
```bash
# Evaluate model performance
python main.py --mode evaluate --model-path models/ppo_model.pth --data-file ../masterreceiver/game_data_log.jsonl
```

#### Status Check
```bash
# Check system status
python main.py --mode status
```

## Model Architecture

### ClashRoyalePPO Network
```python
Input (15 features) → Feature Extractor → LSTM → Attention → Multi-head Output
                                                              ├── Action Policy (5 actions)
                                                              ├── Card Policy (4 slots)  
                                                              ├── Position Policy (x,y)
                                                              ├── Zone Policy (6 zones)
                                                              └── Value Function
```

### Input Features (15D)
1. **Basic**: Elixir count, cards available, troop counts
2. **Spatial**: Enemy/ally troop positions and averages
3. **Strategic**: Troop balance, minimum distances, unique cards
4. **Temporal**: Time since last update
5. **Game State**: Win condition encoding

### Action Space
- **Actions**: wait, place_card, start_match, end_match, defend
- **Cards**: 4 card slots (0-3)
- **Positions**: Continuous x,y coordinates [-1,1] → [200,1200]
- **Zones**: 6 strategic zones (bottom/top + left/center/right)

## Training Process

### Phase 1: Behavioral Cloning (Pre-training)
```bash
python main.py --mode train --epochs 50
```
- Learns from expert gameplay data
- Initializes policy with good strategies
- Faster convergence than random initialization

### Phase 2: PPO Fine-tuning (Reinforcement Learning)
```bash
python main.py --mode play --model-path models/pretrained.pth
```
- Learns through self-play
- Optimizes win rate and resource management
- Continuous improvement through experience

### PPO Hyperparameters
- **Clip Ratio**: 0.2 (prevents large policy updates)
- **Learning Rate**: 3e-4
- **Batch Size**: 64
- **Update Epochs**: 4 per batch
- **GAE Lambda**: 0.95 (advantage estimation)
- **Entropy Coefficient**: 0.01 (exploration bonus)

## Integration with Existing Modules

### Data Flow
```
Game Modules → DataAggregator → PPO Agent → GamePlayer → Actions
     ↓              ↓              ↓           ↓           ↓
  ZeroMQ        Unified State   Policy      Card        Game
  Streams       (JSON)          Network     Placement   Execution
```

### Module Dependencies
- **Required**: publisher, elixir_counter, win_detection
- **Optional**: card_detection, troop_detection
- **Integration**: Seamless with existing ZeroMQ architecture

## Testing and Debugging

### Component Tests
```bash
# Test health checker
python new_integration_layer.py --test-health

# Test data aggregation
python new_integration_layer.py --test-aggregator

# Test game player integration
python new_integration_layer.py --test-player

# Test ZeroMQ connections
python new_integration_layer.py --test-connection
```

### Model Testing
```bash
# Test model loading and inference
python test_model.py --model-path models/ppo_model.pth

# Validate training pipeline
python training_pipeline.py --validate
```

### Debug Mode
```bash
# Run with detailed logging
python main.py --mode play --model-path models/ppo_model.pth --debug

# Monitor decision making
tail -f ai_decisions.log
tail -f ai_decisions.jsonl
```

## Performance Monitoring

### Real-time Metrics
- **Win Rate**: Tracked per session
- **Decision Confidence**: Average policy confidence
- **Action Distribution**: Frequency of different actions
- **Resource Management**: Elixir efficiency

### Logging
- **Structured Decisions**: `ai_decisions.jsonl`
- **Game Data**: `../masterreceiver/game_data_log.jsonl`
- **Training Logs**: `ai_controller.log`

## Advanced Usage

### Custom Training
```python
from clash_royale_ai import ClashRoyalePPOAgent, train_ppo_model

# Custom training loop
agent = ClashRoyalePPOAgent()
agent.run(training=True)

# Pre-training with custom data
model = train_ppo_model("custom_data.jsonl", epochs=200)
```

### Model Evaluation
```python
from training_pipeline import ModelEvaluator

evaluator = ModelEvaluator(model, device)
metrics = evaluator.evaluate(test_loader)
print(f"Win Rate: {metrics['win_rate']:.2%}")
```

### Integration with Other Bots
```python
# Use as a component in larger systems
from clash_royale_ai import ClashRoyalePPO

model = ClashRoyalePPO()
model.load_state_dict(torch.load("ppo_model.pth"))

# Get action probabilities
outputs = model(game_state_tensor)
action_probs = torch.softmax(outputs['action_logits'], dim=-1)
```

## Troubleshooting

### Common Issues

1. **Services Not Healthy**
   ```bash
   python new_integration_layer.py --test-health
   # Check if all required modules are running
   ```

2. **No Training Data**
   ```bash
   python main.py --mode collect-data --required-only
   # Run for 10+ minutes to collect sufficient data
   ```

3. **Model Not Learning**
   - Check reward function in `_calculate_reward()`
   - Verify action execution in `execute_action()`
   - Monitor loss curves in training logs

4. **Poor Performance**
   - Increase training epochs
   - Adjust PPO hyperparameters
   - Collect more diverse training data

### Performance Tips
- Use GPU if available (`CUDA_VISIBLE_DEVICES=0`)
- Increase batch size for stable training
- Monitor entropy to ensure exploration
- Use tensorboard for training visualization

## File Structure
```
deeplearning/
├── clash_royale_ai.py          # Main PPO agent implementation
├── main.py                     # Entry point and orchestration
├── integration_layer.py        # Module management and coordination
├── new_integration_layer.py    # Component testing utilities
├── training_pipeline.py        # Training and evaluation pipeline
├── test_model.py              # Model testing utilities
├── utils/                     # Utility modules
│   ├── data_aggregator.py     # Game state aggregation
│   ├── health_checker.py      # Service health monitoring
│   └── connection_manager.py  # ZeroMQ connection management
├── models/                    # Saved model checkpoints
├── logs/                      # Training and decision logs
└── README.md                  # This file
```

## Contributing

1. **Add New Features**: Extend the PPO model or add new input features
2. **Improve Rewards**: Enhance the reward function for better learning
3. **Optimize Performance**: Profile and optimize bottlenecks
4. **Add Tests**: Expand the testing suite for robustness

## License

This project follows the main repository license. Use responsibly and in accordance with Clash Royale's Terms of Service.