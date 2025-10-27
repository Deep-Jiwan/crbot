#!/usr/bin/env python3
"""
Test Script for Clash Royale AI Model

This script loads a trained model and shows you what it outputs for different inputs.
You can see exactly what the AI "thinks" for various game states.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from clash_royale_ai import ClashRoyaleAI

def load_model(model_path: str):
    """Load a trained model from checkpoint"""
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint first to detect input size
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Try to detect input size from the model weights
    # The LSTM weight shape is [hidden_size*4, input_size]
    lstm_weight_shape = checkpoint['model_state_dict']['lstm.weight_ih_l0'].shape
    input_size = lstm_weight_shape[1]  # Second dimension is input size
    
    print(f"  - Detected input size: {input_size}")
    
    # Create model with correct input size
    model = ClashRoyaleAI(input_size=input_size, hidden_size=128)
    
    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    print("✓ Model loaded successfully!")
    print(f"  - Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  - Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    return model, input_size

def test_single_input(model, elixir, num_cards, num_troops, input_size=3):
    """Test model with a single input state"""
    
    # Create input features based on input_size
    if input_size == 7:
        # Full feature set used by training pipeline
        # [elixir, cards_count, troops_count, win_encoded, unique_cards, unique_troops, troop_ratio]
        features = np.array([
            elixir,           # Elixir count
            num_cards,        # Number of cards
            num_troops,       # Number of troops
            0.0,              # Win condition (0 = ongoing)
            num_cards,        # Unique cards (assume all unique)
            min(num_troops, 3),  # Unique troop types (estimate)
            0.5               # Troop ratio (assume balanced)
        ], dtype=np.float32)
    else:
        # Simple 3-feature set
        features = np.array([elixir, num_cards, num_troops], dtype=np.float32)
    
    # Convert to tensor and add batch + sequence dimensions
    # Shape: (batch=1, sequence_length=1, input_size=3)
    state_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(state_tensor)
    
    # Parse outputs
    action_logits = outputs['action_logits'][0]  # Remove batch dimension
    card_logits = outputs['card_logits'][0]
    position = outputs['position'][0]
    confidence = outputs['confidence'][0, 0].item()
    value = outputs['value'][0, 0].item()
    
    # Convert logits to probabilities
    action_probs = torch.softmax(action_logits, dim=0).numpy()
    card_probs = torch.softmax(card_logits, dim=0).numpy()
    
    # Get best actions
    best_action_idx = torch.argmax(action_logits).item()
    best_card_idx = torch.argmax(card_logits).item()
    
    # Action mapping
    action_map = {
        0: "wait",
        1: "place_card",
        2: "start_match", 
        3: "end_match",
        4: "defend"
    }
    
    # Convert normalized position to pixel coordinates
    target_x = int(position[0].item() * 680 + 200)  # Scale to game area
    target_y = int(position[1].item() * 1000 + 200)
    
    return {
        'action': action_map[best_action_idx],
        'action_idx': best_action_idx,
        'action_probs': action_probs,
        'card_slot': best_card_idx,
        'card_probs': card_probs,
        'target_x': target_x,
        'target_y': target_y,
        'position_raw': position.numpy(),
        'confidence': confidence,
        'value': value
    }

def print_prediction(elixir, num_cards, num_troops, result):
    """Pretty print the prediction results"""
    
    print(f"\n{'='*70}")
    print(f"INPUT: Elixir={elixir}, Cards={num_cards}, Troops={num_troops}")
    print(f"{'='*70}")
    
    print(f"\n🎯 DECISION: {result['action'].upper()}")
    print(f"   Confidence: {result['confidence']*100:.1f}%")
    print(f"   State Value: {result['value']:.3f}")
    
    print(f"\n📊 Action Probabilities:")
    actions = ["wait", "place_card", "start_match", "end_match", "defend"]
    for i, (action, prob) in enumerate(zip(actions, result['action_probs'])):
        bar = "█" * int(prob * 50)
        marker = " ← CHOSEN" if i == result['action_idx'] else ""
        print(f"   {action:12s}: {prob:6.2%} {bar}{marker}")
    
    if result['action'] == 'place_card':
        print(f"\n🃏 Card Selection:")
        for i, prob in enumerate(result['card_probs']):
            bar = "█" * int(prob * 50)
            marker = " ← CHOSEN" if i == result['card_slot'] else ""
            print(f"   Slot {i}: {prob:6.2%} {bar}{marker}")
        
        print(f"\n📍 Placement Position:")
        print(f"   Target: ({result['target_x']}, {result['target_y']}) pixels")
        print(f"   Normalized: ({result['position_raw'][0]:.3f}, {result['position_raw'][1]:.3f})")

def run_test_scenarios(model, input_size):
    """Run multiple test scenarios"""
    
    print("\n" + "="*70)
    print("TESTING DIFFERENT GAME SCENARIOS")
    print("="*70)
    
    scenarios = [
        {
            "name": "Early game - Low elixir",
            "elixir": 2,
            "cards": 4,
            "troops": 0
        },
        {
            "name": "Mid game - Good elixir",
            "elixir": 7,
            "cards": 4,
            "troops": 2
        },
        {
            "name": "Full elixir - Should act",
            "elixir": 10,
            "cards": 4,
            "troops": 1
        },
        {
            "name": "Low elixir - Many troops",
            "elixir": 1,
            "cards": 3,
            "troops": 5
        },
        {
            "name": "High elixir - No troops",
            "elixir": 8,
            "cards": 4,
            "troops": 0
        }
    ]
    
    for scenario in scenarios:
        result = test_single_input(
            model,
            scenario['elixir'],
            scenario['cards'],
            scenario['troops'],
            input_size
        )
        print_prediction(
            scenario['elixir'],
            scenario['cards'],
            scenario['troops'],
            result
        )
        input("\nPress Enter to see next scenario...")

def interactive_mode(model, input_size):
    """Interactive mode where you can input values"""
    
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter game state values to see what the AI would do.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            elixir = input("Elixir (0-10): ").strip()
            if elixir.lower() == 'quit':
                break
            elixir = int(elixir)
            
            cards = input("Number of cards in hand (0-4): ").strip()
            if cards.lower() == 'quit':
                break
            cards = int(cards)
            
            troops = input("Number of troops on field (0-20): ").strip()
            if troops.lower() == 'quit':
                break
            troops = int(troops)
            
            result = test_single_input(model, elixir, cards, troops, input_size)
            print_prediction(elixir, cards, troops, result)
            
            print("\n" + "-"*70 + "\n")
            
        except ValueError as e:
            print(f"Invalid input: {e}")
        except KeyboardInterrupt:
            break
    
    print("\nExiting interactive mode.")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Clash Royale AI Model")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--mode", choices=["test", "interactive"], default="test",
                       help="Mode: test scenarios or interactive input")
    parser.add_argument("--elixir", type=int, help="Elixir count for single test")
    parser.add_argument("--cards", type=int, help="Number of cards for single test")
    parser.add_argument("--troops", type=int, help="Number of troops for single test")
    
    args = parser.parse_args()
    
    # Check if model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("\nAvailable models:")
        models_dir = Path("models")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pth"):
                print(f"  - {model_file}")
        else:
            print("  - No models directory found")
        return
    
    # Load model
    model, input_size = load_model(str(model_path))
    
    # Run based on mode
    if args.elixir is not None and args.cards is not None and args.troops is not None:
        # Single test
        print("\nSINGLE TEST MODE")
        result = test_single_input(model, args.elixir, args.cards, args.troops, input_size)
        print_prediction(args.elixir, args.cards, args.troops, result)
    elif args.mode == "interactive":
        interactive_mode(model, input_size)
    else:
        run_test_scenarios(model, input_size)

if __name__ == "__main__":
    main()
