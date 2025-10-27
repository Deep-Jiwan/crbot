"""
Example: Using the Health Checker in other Python applications

This demonstrates various ways to use the health_checker module in your own scripts.
"""

from utils.health_checker import HealthChecker, ServiceType
import sys


def example_basic_check():
    """Example 1: Basic health check"""
    print("="*60)
    print("EXAMPLE 1: Basic Health Check")
    print("="*60)
    
    checker = HealthChecker()
    results = checker.check_all_services()
    
    if checker.are_all_required_services_healthy(results):
        print("✓ Ready to start!")
        return True
    else:
        print("✗ Not ready - some services are down")
        return False


def example_wait_for_services():
    """Example 2: Wait for services to become ready"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Wait for Services")
    print("="*60)
    
    checker = HealthChecker()
    
    # Wait up to 30 seconds for services to be ready
    if checker.wait_for_services(timeout_sec=30):
        print("✓ All services are ready!")
        return True
    else:
        print("✗ Timeout - services didn't start in time")
        return False


def example_check_specific_service():
    """Example 3: Check a specific service"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Check Specific Service")
    print("="*60)
    
    checker = HealthChecker()
    
    # Check just the elixir service
    result = checker.check_service("elixir")
    
    if result:
        print(f"Service: {result.service_name}")
        print(f"Status: {'Healthy' if result.is_healthy else 'Unhealthy'}")
        print(f"Response Time: {result.response_time_ms:.1f}ms")
        if result.error_message:
            print(f"Error: {result.error_message}")
        return result.is_healthy
    else:
        print("Service 'elixir' not found")
        return False


def example_add_custom_service():
    """Example 4: Add and check custom services"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Add Custom Service")
    print("="*60)
    
    checker = HealthChecker()
    
    # Add a custom HTTP service
    checker.add_service(
        name="my_api",
        service_type=ServiceType.HTTP,
        port_or_url="http://localhost:8000/health",
        required=False,
        timeout_ms=2000
    )
    
    # Check it
    result = checker.check_service("my_api")
    if result:
        print(f"Custom service check: {'✓' if result.is_healthy else '✗'}")
    else:
        print("Could not check custom service")


def example_get_unhealthy_services():
    """Example 5: Get list of unhealthy services"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Get Unhealthy Services")
    print("="*60)
    
    checker = HealthChecker()
    results = checker.check_all_services(verbose=False)
    
    unhealthy = checker.get_unhealthy_services(results)
    
    if unhealthy:
        print(f"Unhealthy services: {', '.join(unhealthy)}")
    else:
        print("All services are healthy!")


def example_continuous_monitoring():
    """Example 6: Continuous monitoring (runs for 20 seconds)"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Continuous Monitoring (20 seconds)")
    print("="*60)
    
    import time
    checker = HealthChecker()
    
    print("Monitoring services every 5 seconds...")
    for i in range(4):  # 4 checks = 20 seconds
        print(f"\n--- Check {i+1} ---")
        results = checker.check_all_services(verbose=True)
        
        if i < 3:  # Don't sleep after last check
            time.sleep(5)


def example_check_required_only():
    """Example 7: Check only required services"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Check Required Services Only")
    print("="*60)
    
    checker = HealthChecker()
    results = checker.check_all_services(required_only=True)
    
    all_healthy = checker.are_all_required_services_healthy(results)
    print(f"\nAll required services healthy: {all_healthy}")


def example_integration_with_app():
    """Example 8: Typical integration pattern in an application"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Application Integration Pattern")
    print("="*60)
    
    checker = HealthChecker()
    
    # 1. Check services at startup
    print("\n1. Checking services at startup...")
    if not checker.are_all_required_services_healthy(
        checker.check_all_services(verbose=False)
    ):
        print("   Waiting for services to start...")
        if not checker.wait_for_services(timeout_sec=30):
            print("   ERROR: Required services not available")
            sys.exit(1)
    
    print("   ✓ Services ready")
    
    # 2. Your application logic here
    print("\n2. Running application logic...")
    print("   (your code would go here)")
    
    # 3. Optional: Periodic health checks during runtime
    print("\n3. Periodic health check...")
    results = checker.check_all_services(verbose=False)
    if not checker.are_all_required_services_healthy(results):
        print("   WARNING: Some services became unhealthy!")
        unhealthy = checker.get_unhealthy_services(results)
        print(f"   Unhealthy: {', '.join(unhealthy)}")
    else:
        print("   ✓ All services still healthy")


def main():
    """Run all examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Health Checker Examples")
    parser.add_argument("--example", type=int, choices=range(1, 9),
                       help="Run a specific example (1-8)")
    args = parser.parse_args()
    
    examples = {
        1: example_basic_check,
        2: example_wait_for_services,
        3: example_check_specific_service,
        4: example_add_custom_service,
        5: example_get_unhealthy_services,
        6: example_continuous_monitoring,
        7: example_check_required_only,
        8: example_integration_with_app,
    }
    
    if args.example:
        # Run specific example
        examples[args.example]()
    else:
        # Run all examples (except continuous monitoring)
        print("\n" + "="*60)
        print("RUNNING ALL EXAMPLES")
        print("="*60)
        
        example_basic_check()
        # example_wait_for_services()  # Skip - might take 30 seconds
        example_check_specific_service()
        example_add_custom_service()
        example_get_unhealthy_services()
        # example_continuous_monitoring()  # Skip - takes 20 seconds
        example_check_required_only()
        example_integration_with_app()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETE")
        print("="*60)
        print("\nTip: Run with --example N to run a specific example")
        print("     Example: python health_checker_examples.py --example 2")


if __name__ == "__main__":
    main()
