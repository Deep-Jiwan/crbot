# Enhanced Clash Royale Deep Learning AI

A comprehensive deep learning system that integrates with all existing Clash Royale bot modules to provide intelligent gameplay automation with enhanced features.

## Overview

This enhanced deep learning module creates an AI agent that can:
- Process comprehensive game data (4 cards, enemy/ally troop locations, game state)
- Make intelligent decisions with standardized JSON output format
- Provide human-readable reasoning for all decisions
- Use zone-based position prediction (bottom-left, bottom-center, etc.)
- Learn from gameplay experience using reinforcement learning
- Adapt strategies based on game state and opponent behavior
- Log all decisions in structured format for analysis

## Architecture

### Core Components

1. **ClashRoyaleAI** - Main neural network model with LSTM and attention mechanisms
2. **IntegrationLayer** - Manages all existing modules and data flow
3. **TrainingPipeline** - Comprehensive training and evaluation system
4. **AIController** - Main orchestrator for the entire system

### Data Flow

```
Publisher → Elixir Counter → Master Receiver → AI Agent → Game Player
    ↓           ↓              ↓                ↓
Card Detection → Data Aggregator → Decision Engine → Action Execution
    ↓
Troop Detection → Win Detection
```

## Features

### Enhanced Deep Learning Model
- **2-layer LSTM architecture** for advanced sequence processing
- **Multi-head attention** for focusing on important game states
- **Enhanced input processing** with 15 comprehensive features
- **Zone-based position prediction** (6 predefined zones)
- **Reinforcement learning** with experience replay
- **Real-time decision making** with confidence scoring and reasoning
- **Structured JSON output** for all decisions
- **Comprehensive logging** for analysis and debugging

### Integration
- **Unified interface** to all existing modules
- **Automatic module management** with health monitoring
- **Data aggregation** from multiple sources
- **Error handling and recovery**

### Training System
- **Comprehensive data preprocessing**
- **Multiple training strategies** (supervised, reinforcement)
- **Hyperparameter tuning** and optimization
- **Model evaluation** with detailed metrics
- **Checkpointing** and model versioning

## Installation

1. **Setup the environment:**
   ```bash
   cd deeplearning
   python setup.py
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   - Edit `.env` file with your settings
   - Set up Roboflow API keys if using computer vision modules
   - Adjust game coordinates for your setup

## Usage

### Quick Start

1. **Check system status:**
   ```bash
   python main.py --mode status
   ```

2. **Collect training data:**
   ```bash
   python main.py --mode collect-data --required-only
   ```

3. **Train the AI model:**
   ```bash
   python main.py --mode train --epochs 100
   ```

4. **Play with AI:**
   ```bash
   python main.py --mode play --model-path models/best_model.pth
   ```

### Detailed Usage

#### Training Mode
```bash
python main.py --mode train \
    --data-file ../masterreceiver/game_data_log.jsonl \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001
```

#### Play Mode
```bash
python main.py --mode play \
    --model-path models/best_model.pth \
    --required-only
```

#### Data Collection Mode
```bash
python main.py --mode collect-data \
    --required-only
```

#### Evaluation Mode
```bash
python main.py --mode evaluate \
    --model-path models/best_model.pth \
    --data-file ../masterreceiver/game_data_log.jsonl
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```env
# Frame dimensions
FRAME_WIDTH=1080
FRAME_HEIGHT=1920

# ZeroMQ ports
ZMQ_ADDRESS=tcp://localhost:5550
ELIXIR_PORT=5551
CARDS_PORT=5552
TROOPS_PORT=5560
WIN_PORT=5570

# Training configuration
BATCH_SIZE=32
LEARNING_RATE=0.001
EPOCHS=100

# Enhanced Model configuration
HIDDEN_SIZE=256
INPUT_SIZE=15
SEQUENCE_LENGTH=10
```

### Game Coordinates

Adjust these coordinates for your specific setup:

```env
# Card slots (left to right)
CARD_0_X=225
CARD_0_Y=1560
CARD_1_X=415
CARD_1_Y=1560
CARD_2_X=605
CARD_2_Y=1560
CARD_3_X=795
CARD_3_Y=1560

# Match control
START_MATCH_X=540
START_MATCH_Y=1200
END_MATCH_X=540
END_MATCH_Y=1400
```

## Model Architecture

### Enhanced Neural Network Design

```
Input Layer (15 Enhanced Features)
    ↓
Input Encoder (Linear + ReLU + Dropout)
    ↓
2-Layer LSTM (256 hidden units each)
    ↓
Multi-Head Attention (8 heads)
    ↓
Enhanced Dense Layers
    ├── Action Head (5 actions)
    ├── Card Head (4 card slots)
    ├── Position Head (x, y coordinates)
    ├── Zone Head (6 zones)
    ├── Confidence Head (action confidence)
    └── Value Head (state value)
```

### Training Strategy

1. **Supervised Learning** - Learn from historical game data
2. **Reinforcement Learning** - Learn from real-time gameplay
3. **Experience Replay** - Store and replay past experiences
4. **Target Network** - Stable learning with target network updates

## Data Format

### Enhanced Input Features (15 features)
- **Basic game state**: Elixir count, cards in hand, troops on field
- **Enemy troop analysis**: Count, average position (x, y)
- **Ally troop analysis**: Count, average position (x, y)
- **Strategic features**: Unique cards, win condition, troop balance
- **Temporal features**: Distance between troops, time since update

### Enhanced Output Actions
- `wait` - No action (with reasoning)
- `place_card` - Place a card with zone prediction
- `start_match` - Start a match
- `end_match` - End/surrender match
- `defend` - Defensive action

### Zone-based Position Prediction
- `bottom_left` - Bottom left area
- `bottom_center` - Bottom center area (most common)
- `bottom_right` - Bottom right area
- `top_left` - Top left area
- `top_center` - Top center area
- `top_right` - Top right area

## Monitoring and Logging

### Enhanced Logging
- **AI Controller**: `ai_controller.log`
- **AI Decisions**: `ai_decisions.jsonl` (structured JSON format)
- **AI Decisions**: `ai_decisions.log` (human-readable format)
- **Training**: TensorBoard logs in `logs/`
- **Module Status**: Real-time status updates

### Enhanced Metrics
- **Training Loss**: MSE loss during training
- **Validation Loss**: Performance on validation set
- **Reward**: Game performance metrics
- **Accuracy**: Action prediction accuracy
- **Decision Confidence**: AI confidence scores
- **Reasoning Quality**: Human-readable decision explanations

## Troubleshooting

### Common Issues

1. **Module startup failures:**
   - Check if all required modules are installed
   - Verify environment variables are set correctly
   - Ensure ports are not already in use

2. **Training issues:**
   - Check if training data exists and is valid
   - Verify GPU availability for training
   - Monitor memory usage during training

3. **AI decision making:**
   - Ensure model is properly trained
   - Check game coordinates are correct
   - Verify data flow from modules

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Development

### Adding New Features

1. **New input features**: Modify `DataPreprocessor._extract_features()`
2. **New actions**: Update `action_map` in `ClashRoyaleAgent`
3. **New modules**: Add to `ModuleManager.module_configs`

### Testing

Run tests:
```bash
python -m pytest tests/
```

### Code Quality

Pre-commit hooks are automatically installed:
- Python syntax checking
- Import validation
- Code formatting

## Performance

### Optimization Tips

1. **Use GPU** for training when available
2. **Adjust batch size** based on available memory
3. **Tune sequence length** for your use case
4. **Monitor model complexity** vs performance

### Enhanced Benchmarks

Typical performance on modern hardware:
- **Training**: ~100 epochs in 2-3 hours (enhanced features)
- **Inference**: <15ms per decision (with reasoning generation)
- **Memory**: ~3GB for training, ~800MB for inference (enhanced model)
- **Decision Logging**: <1ms per decision (JSON format)

## System Architecture

### File Structure

```
deeplearning/
├── clash_royale_ai.py      # Enhanced AI model with 15 features, zone prediction, JSON output
├── integration_layer.py    # Module management and data aggregation
├── training_pipeline.py    # Enhanced training with 15 features, 256 hidden units
├── main.py                 # Unified controller for all modes
├── requirements.txt        # Enhanced dependencies
├── setup.py               # Environment setup with enhanced configuration
└── README.md              # This comprehensive documentation
```

### Enhanced Features in Action

- **Input**: 15 comprehensive features including enemy/ally troop positions
- **Output**: JSON format with reasoning: `"Playing Knight at bottom_center zone to counter enemy advantage"`
- **Logging**: All decisions saved in `ai_decisions.jsonl` and `ai_decisions.log`
- **Positioning**: Zone-based instead of raw coordinates (more strategic)

### Benefits of Enhanced System

1. **No Version Confusion** - Single enhanced system
2. **Better Performance** - 15 features vs 3, 256 hidden units vs 128
3. **Transparent AI** - Every decision explained in human language
4. **Easy Integration** - Standardized JSON output format
5. **Comprehensive Logging** - Full decision analysis and debugging
6. **Strategic Positioning** - Zone-based instead of raw coordinates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the Clash Royale Bot system. See the main repository for license information.

## Disclaimer

This project is not affiliated with Supercell. Use at your own risk—automated gameplay may violate Clash Royale's Terms of Service.
