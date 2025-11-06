#!/usr/bin/env python3
"""
Simple test script for PPO model
"""

import torch
import numpy as np
from clash_royale_ai import ClashRoyalePPO, ClashRoyalePPOAgent

def test_model_forward():
    """Test model forward pass"""
    print("Testing PPO model...")
    
    model = ClashRoyalePPO(input_size=15, hidden_size=256, num_actions=3)
    input_tensor = torch.randn(2, 1, 15)
    outputs = model(input_tensor)
    
    assert outputs['action_logits'].shape == (2, 3)
    assert outputs['card_logits'].shape == (2, 4)
    assert outputs['zone_logits'].shape == (2, 6)
    assert outputs['value'].shape == (2, 1)
    
    print("✓ Forward pass test passed")

def test_action_sampling():
    """Test action sampling"""
    print("Testing action sampling...")
    
    model = ClashRoyalePPO(input_size=15, hidden_size=256, num_actions=3)
    input_tensor = torch.randn(1, 1, 15)
    result = model.get_action_and_value(input_tensor)
    
    assert 0 <= result['action'].item() < 3
    assert 0 <= result['card'].item() < 4
    assert 0 <= result['zone'].item() < 6
    assert result['position'].shape == (1, 2)
    
    print("✓ Action sampling test passed")

def test_agent_initialization():
    """Test agent initialization"""
    print("Testing agent initialization...")
    
    try:
        agent = ClashRoyalePPOAgent()
        print("✓ Agent initialization test passed")
        return agent
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        return None

def test_feature_extraction():
    """Test feature extraction"""
    print("Testing feature extraction...")
    
    agent = ClashRoyalePPOAgent()
    
    # Set dummy game state
    agent.current_state.elixir = 5
    agent.current_state.cards_in_hand = [
        {"slot": 0, "name": "Fireball"},
        {"slot": 1, "name": "Knight"}
    ]
    agent.current_state.enemy_troops = [
        {"type": "Goblin", "x": 400, "y": 300, "team": "enemy"}
    ]
    agent.current_state.ally_troops = [
        {"type": "Knight", "x": 500, "y": 800, "team": "ally"}
    ]
    
    features = agent.get_state_features()
    
    assert len(features) == 15
    assert features[0] == 5  # elixir
    assert features[1] == 2  # cards count
    
    print("✓ Feature extraction test passed")

def main():
    """Run all tests"""
    print("=" * 50)
    print("PPO MODEL TESTING")
    print("=" * 50)
    
    tests = [
        test_model_forward,
        test_action_sampling,
        test_agent_initialization,
        test_feature_extraction
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
        print()
    
    print("=" * 50)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 50)

if __name__ == "__main__":
    main()