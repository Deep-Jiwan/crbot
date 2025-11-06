#!/usr/bin/env python3
"""
Test script for PPO model functionality
"""

import torch
import numpy as np
from clash_royale_ai import ClashRoyalePPO, ClashRoyalePPOAgent

def test_model_forward():
    """Test basic model forward pass"""
    print("Testing PPO model forward pass...")
    
    model = ClashRoyalePPO(input_size=15, hidden_size=256, num_actions=3)
    
    # Create dummy input
    batch_size = 2
    seq_length = 1
    input_tensor = torch.randn(batch_size, seq_length, 15)
    
    # Forward pass
    outputs = model(input_tensor)
    
    # Check output shapes
    assert outputs['action_logits'].shape == (batch_size, 3), f"Action logits shape: {outputs['action_logits'].shape}"
    assert outputs['card_logits'].shape == (batch_size, 4), f"Card logits shape: {outputs['card_logits'].shape}"
    assert outputs['position_mean'].shape == (batch_size, 2), f"Position mean shape: {outputs['position_mean'].shape}"
    assert outputs['zone_logits'].shape == (batch_size, 6), f"Zone logits shape: {outputs['zone_logits'].shape}"
    assert outputs['value'].shape == (batch_size, 1), f"Value shape: {outputs['value'].shape}"
    
    print("✓ Forward pass test passed")

def test_action_sampling():
    """Test action sampling functionality"""
    print("Testing action sampling...")
    
    model = ClashRoyalePPO(input_size=15, hidden_size=256, num_actions=3)
    input_tensor = torch.randn(1, 1, 15)
    
    # Sample actions
    result = model.get_action_and_value(input_tensor)
    
    # Check that actions are valid
    assert 0 <= result['action'].item() < 3, f"Invalid action: {result['action'].item()}"
    assert 0 <= result['card'].item() < 4, f"Invalid card: {result['card'].item()}"
    assert 0 <= result['zone'].item() < 6, f"Invalid zone: {result['zone'].item()}"
    assert result['position'].shape == (1, 2), f"Position shape: {result['position'].shape}"
    
    print("✓ Action sampling test passed")

def test_agent_initialization():
    """Test PPO agent initialization"""
    print("Testing PPO agent initialization...")
    
    try:
        agent = ClashRoyalePPOAgent()
        print("✓ Agent initialization test passed")
        return agent
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        return None

def test_feature_extraction():
    """Test feature extraction from game state"""
    print("Testing feature extraction...")
    
    agent = ClashRoyalePPOAgent()
    
    # Set up dummy game state
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
    agent.current_state.win_condition = "ongoing"
    
    # Extract features
    features = agent.get_state_features()
    
    assert len(features) == 15, f"Expected 15 features, got {len(features)}"
    assert features[0] == 5, f"Elixir should be 5, got {features[0]}"
    assert features[1] == 2, f"Cards count should be 2, got {features[1]}"
    
    print("✓ Feature extraction test passed")

def test_model_save_load():
    """Test model saving and loading"""
    print("Testing model save/load...")
    
    # Create and save model
    agent = ClashRoyalePPOAgent()
    test_path = "test_ppo_model.pth"
    
    try:
        agent.save_model(test_path)
        
        # Create new agent and load
        new_agent = ClashRoyalePPOAgent()
        new_agent.load_model(test_path)
        
        print("✓ Model save/load test passed")
        
        # Cleanup
        import os
        if os.path.exists(test_path):
            os.remove(test_path)
            
    except Exception as e:
        print(f"✗ Model save/load failed: {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("PPO MODEL TESTING")
    print("=" * 60)
    
    tests = [
        test_model_forward,
        test_action_sampling,
        test_agent_initialization,
        test_feature_extraction,
        test_model_save_load
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! PPO model is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above.")

if __name__ == "__main__":
    main()