#!/usr/bin/env python3
"""
AIO Services Main Orchestrator
Runs all 4 microservices in a single container with separate threads
"""
import sys
import threading
import time
import zmq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all services
sys.path.insert(0, 'services')
from winwin.winwin_service import run_winwin_service
from elixircount.elixir_service import run_elixir_service
from carddetection.card_service import run_card_service
from troopdetection.troop_service import run_troop_service

def run_service_in_thread(service_func, service_name, context):
    """Wrapper to run a service in a thread with error handling"""
    try:
        print(f"[MAIN] Starting {service_name}...")
        service_func(context)
    except Exception as e:
        print(f"[MAIN] ERROR in {service_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main orchestrator function"""
    print("=" * 70)
    print("AIO Services Container - All-In-One Microservices")
    print("=" * 70)
    print("[MAIN] Starting all services...")
    print("[MAIN] Services: WinWin (5570), Elixir (5551), Cards (5554), Troops (5560)")
    print("=" * 70)
    
    # Create shared ZMQ context for all services
    context = zmq.Context()
    
    # Create threads for each service
    services = [
        ("WinWin Detection", run_winwin_service),
        ("Elixir Counter", run_elixir_service),
        ("Card Detection", run_card_service),
        ("Troop Detection", run_troop_service),
    ]
    
    threads = []
    for service_name, service_func in services:
        thread = threading.Thread(
            target=run_service_in_thread,
            args=(service_func, service_name, context),
            daemon=True,
            name=service_name
        )
        thread.start()
        threads.append(thread)
        # Small delay between starting services to avoid initialization conflicts
        time.sleep(0.5)
    
    print("[MAIN] All services started successfully!")
    print("[MAIN] Press Ctrl+C to stop all services")
    
    # Keep main thread alive
    try:
        while True:
            # Check if any thread has died
            for i, thread in enumerate(threads):
                if not thread.is_alive():
                    service_name = services[i][0]
                    print(f"[MAIN] WARNING: {service_name} thread has died!")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down all services...")
        context.term()
        print("[MAIN] Goodbye!")

if __name__ == "__main__":
    main()
