"""
Simple Integration Layer for Clash Royale AI
Checks if services are running, connects to them, and provides access to game state.
"""

import sys
import os
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

# Import utils modules
from utils.health_checker import HealthChecker
from utils.connection_manager import ConnectionManager
from utils.data_aggregator import DataAggregator

# Load environment variables
load_dotenv()

# Add gameplayer to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gameplayer"))
from gameplayer import GamePlayer


def test_health_checker():
    """Test the health checker module"""
    print("\n" + "="*60)
    print("TESTING HEALTH CHECKER")
    print("="*60)
    
    checker = HealthChecker()
    results = checker.check_all_services()
    
    all_healthy = checker.are_all_required_services_healthy(results)
    print(f"\nAll required services healthy: {all_healthy}")
    
    return all_healthy


def test_connection_manager():
    """Test the connection manager module"""
    print("\n" + "="*60)
    print("TESTING CONNECTION MANAGER")
    print("="*60)
    
    manager = ConnectionManager()
    
    # Show service info
    print("\nConfigured services:")
    for key, info in manager.get_service_info().items():
        print(f"  - {info['name']}: {info['address']}")
    
    # Try to connect
    print("\nConnecting to services...")
    if manager.connect():
        print("✓ Connected successfully")
        
        # Try to receive a message
        print("\nTrying to receive messages (5 second timeout)...")
        start_time = time.time()
        message_count = 0
        
        while time.time() - start_time < 5:
            msg = manager.receive_message()
            if msg:
                topic, data = msg
                message_count += 1
                data_preview = data[:50] + "..." if len(data) > 50 else data
                print(f"  [{topic}] {data_preview}")
        
        print(f"\nReceived {message_count} messages")
        manager.disconnect()
        return True
    else:
        print("✗ Connection failed")
        return False


def test_data_aggregator():
    """Test the data aggregator module"""
    print("\n" + "="*60)
    print("TESTING DATA AGGREGATOR")
    print("="*60)
    
    aggregator = DataAggregator()
    aggregator.start()
    
    print("\nMonitoring data for 10 seconds...")
    start_time = time.time()
    
    try:
        while time.time() - start_time < 10:
            # Get data as JSON string, then parse it
            json_data = aggregator.get_current_data_json()
            data = aggregator.get_current_data()
            
            print(f"\r[{int(time.time() - start_time)}s] "
                  f"Elixir: {data['elixir']:2d} | "
                  f"Cards: {len(data['cards'])} | "
                  f"Troops: {len(data['troops'])} | "
                  f"Win: {data['win_condition']:<10}", end="")
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    
    print("\n")
    aggregator.cleanup()
    return True


def test_game_player():
    """Test the game player module"""
    print("\n" + "="*60)
    print("TESTING GAME PLAYER")
    print("="*60)
    
    try:
        player = GamePlayer()
        print("✓ GamePlayer initialized")
        
        print("\nRunning GamePlayer test functionality...")
        player.test_functionality()
        
        return True
    except Exception as e:
        print(f"✗ GamePlayer test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Clash Royale AI Integration - Module Testing")
    parser.add_argument("--test-health", action="store_true",
                       help="Test health checker only")
    parser.add_argument("--test-connection", action="store_true",
                       help="Test connection manager only")
    parser.add_argument("--test-aggregator", action="store_true",
                       help="Test data aggregator only")
    parser.add_argument("--test-player", action="store_true",
                       help="Test game player only")
    parser.add_argument("--test-all", action="store_true",
                       help="Test all modules")
    
    args = parser.parse_args()
    
    # If no specific test selected, show usage
    if not any([args.test_health, args.test_connection, args.test_aggregator, 
                args.test_player, args.test_all]):
        parser.print_help()
        print("\n" + "="*60)
        print("AVAILABLE TESTS")
        print("="*60)
        print("  --test-health      : Test health checker")
        print("  --test-connection  : Test connection manager")
        print("  --test-aggregator  : Test data aggregator")
        print("  --test-player      : Test game player")
        print("  --test-all         : Run all tests")
        print("\nExample: python new_integration_layer.py --test-all")
        return
    
    results = {}
    
    try:
        if args.test_all or args.test_health:
            results['health'] = test_health_checker()
        
        if args.test_all or args.test_connection:
            results['connection'] = test_connection_manager()
        
        if args.test_all or args.test_aggregator:
            results['aggregator'] = test_data_aggregator()
        
        if args.test_all or args.test_player:
            results['player'] = test_game_player()
        
        # Print summary
        if results:
            print("\n" + "="*60)
            print("TEST SUMMARY")
            print("="*60)
            for test_name, passed in results.items():
                status = "✓ PASSED" if passed else "✗ FAILED"
                print(f"  {test_name.title()}: {status}")
            print("="*60)
    
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    
    except Exception as e:
        print(f"\n\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
