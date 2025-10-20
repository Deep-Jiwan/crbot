import os
import sys
import time
import threading
import runpy
from pathlib import Path

def run_master_receiver():
    """Execute the existing master_receiver.py as a script in this thread."""
    script_path = Path(__file__).parent / "master_receiver.py"
    if not script_path.exists():
        print(f"master_receiver.py not found at {script_path}")
        return
    print("Starting master_receiver in thread")
    try:
        # run as __main__ so module-level code executes as when run directly
        runpy.run_path(str(script_path), run_name="__main__")
    except Exception as e:
        print("master_receiver thread exited with error:", e)


def run_server():
    """Import server module and run uvicorn programmatically."""
    try:
        import uvicorn
    except Exception as e:
        print("uvicorn is required to run the server. Install it in the environment.", e)
        return

    try:
        # Import the server module (this defines `app`)
        import server
    except Exception as e:
        print("Failed to import server module:", e)
        return

    port = int(os.environ.get('PORT', os.environ.get('MASTERRECEIVER_PORT', 8002)))
    print(f"Starting FastAPI server on 0.0.0.0:{port} in thread")
    try:
        # uvicorn.run is blocking; run it inside this thread
        uvicorn.run(server.app, host='0.0.0.0', port=port, log_level='info')
    except Exception as e:
        print("Server thread exited with error:", e)


def main():
    # Start both components in threads
    mr_thread = threading.Thread(target=run_master_receiver, name='master_receiver')
    srv_thread = threading.Thread(target=run_server, name='server')

    # Make threads daemonic so they won't block process exit on KeyboardInterrupt
    mr_thread.daemon = True
    srv_thread.daemon = True

    mr_thread.start()
    # small stagger so logs don't mix too much
    time.sleep(0.2)
    srv_thread.start()

    print("Both threads started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.5)
            # if both threads have finished, exit
            if not mr_thread.is_alive() and not srv_thread.is_alive():
                print("Both threads exited. Shutting down.")
                break
    except KeyboardInterrupt:
        print("Keyboard interrupt received — shutting down.")
        # Allow daemon threads to exit with the process
        sys.exit(0)


if __name__ == '__main__':
    main()
