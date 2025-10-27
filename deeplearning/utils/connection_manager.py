"""
Connection Manager for Clash Royale AI

This module manages ZMQ connections to all bot services (publisher, elixir counter,
card detection, troop detection, win detection) and provides methods to check
service availability and receive data from subscribed topics.
"""

import os
import zmq
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ServiceConfig:
    """Configuration for a ZMQ service"""
    name: str
    port: int
    topics: List[str]
    required: bool = True
    
    def get_address(self) -> str:
        """Get the TCP address for this service"""
        return f"tcp://localhost:{self.port}"


class ConnectionManager:
    """Manages ZMQ connections to all Clash Royale bot services"""
    
    def __init__(self, timeout_ms: int = 100):
        """
        Initialize the ConnectionManager.
        
        Args:
            timeout_ms: Timeout for ZMQ socket receive in milliseconds
        """
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.subscriber = None
        self.connected = False
        
        # Service configurations
        self.services = {
            "elixir": ServiceConfig(
                name="Elixir Counter",
                port=int(os.getenv("ELIXIR_PORT", "5560")),
                topics=["ecount"],  # Elixir count topic
                required=True
            ),
            "cards": ServiceConfig(
                name="Card Detection",
                port=int(os.getenv("CARDS_PORT", "5590")),
                topics=["cards"],  # Card detection topic
                required=False
            ),
            "troops": ServiceConfig(
                name="Troop Detection",
                port=int(os.getenv("TROOPS_PORT", "5580")),
                topics=["troops"],  # Troop detection topic
                required=False
            ),
            "win": ServiceConfig(
                name="Win Detection",
                port=int(os.getenv("WIN_PORT", "5570")),
                topics=["winner"],  # Win/loss detection topic
                required=True
            )
        }
    
    def check_service(self, service_key: str, timeout: int = 1000) -> bool:
        """
        Check if a specific service is running.
        
        Args:
            service_key: Key of the service to check (e.g., 'elixir', 'cards')
            timeout: Timeout in milliseconds
            
        Returns:
            True if service is responding, False otherwise
        """
        if service_key not in self.services:
            return False
        
        service = self.services[service_key]
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVTIMEO, timeout)
        
        try:
            socket.connect(service.get_address())
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            socket.recv()
            return True
        except zmq.Again:
            # Timeout - no data received
            return False
        except Exception as e:
            print(f"Error checking {service.name}: {e}")
            return False
        finally:
            socket.close()
            context.term()
    
    def check_all_services(self, required_only: bool = False) -> Dict[str, bool]:
        """
        Check all services and return their status.
        
        Args:
            required_only: If True, only check required services
            
        Returns:
            Dictionary mapping service keys to their status (True/False)
        """
        results = {}
        print("\nChecking services...")
        
        for key, service in self.services.items():
            if required_only and not service.required:
                continue
            
            status = self.check_service(key)
            results[key] = status
            
            mark = "✓" if status else "✗"
            req = "(required)" if service.required else "(optional)"
            print(f"  {mark} {service.name} on port {service.port} {req}")
        
        return results
    
    def are_required_services_running(self) -> bool:
        """
        Check if all required services are running.
        
        Returns:
            True if all required services are running, False otherwise
        """
        results = self.check_all_services(required_only=True)
        return all(results.values())
    
    def connect(self, services: Optional[List[str]] = None, 
                subscribe_to_all: bool = True) -> bool:
        """
        Connect to specified services and subscribe to their topics.
        
        Args:
            services: List of service keys to connect to. If None, connects to all.
            subscribe_to_all: If True, subscribes to all topics. If False, subscribe manually.
            
        Returns:
            True if connection successful, False otherwise
        """
        if self.connected:
            print("Already connected. Disconnect first.")
            return False
        
        try:
            # Create subscriber socket
            self.subscriber = self.context.socket(zmq.SUB)
            self.subscriber.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            
            # Connect to specified services
            services_to_connect = services if services else list(self.services.keys())
            
            for service_key in services_to_connect:
                if service_key not in self.services:
                    print(f"Warning: Unknown service '{service_key}'")
                    continue
                
                service = self.services[service_key]
                address = service.get_address()
                self.subscriber.connect(address)
                print(f"Connected to {service.name} at {address}")
                
                # Subscribe to topics
                if subscribe_to_all:
                    for topic in service.topics:
                        self.subscribe_to_topic(topic)
            
            self.connected = True
            print("Connection manager initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to connect: {e}")
            if self.subscriber:
                self.subscriber.close()
                self.subscriber = None
            return False
    
    def subscribe_to_topic(self, topic: str):
        """
        Subscribe to a specific topic.
        
        Args:
            topic: Topic name to subscribe to
        """
        if not self.subscriber:
            print("Not connected. Call connect() first.")
            return
        
        topic_bytes = f"{topic}|".encode()
        self.subscriber.setsockopt(zmq.SUBSCRIBE, topic_bytes)
        print(f"Subscribed to topic: {topic}")
    
    def subscribe_to_all_topics(self):
        """Subscribe to all available topics from all services"""
        if not self.subscriber:
            print("Not connected. Call connect() first.")
            return
        
        for service in self.services.values():
            for topic in service.topics:
                self.subscribe_to_topic(topic)
    
    def receive_message(self, blocking: bool = False) -> Optional[Tuple[str, str]]:
        """
        Receive a message from any subscribed topic.
        
        Args:
            blocking: If True, blocks until message received. If False, returns None on timeout.
            
        Returns:
            Tuple of (topic, data) or None if no message available
        """
        if not self.connected or not self.subscriber:
            return None
        
        try:
            # Temporarily set blocking mode if requested
            if blocking:
                self.subscriber.setsockopt(zmq.RCVTIMEO, -1)
            
            msg = self.subscriber.recv()
            
            # Restore timeout
            if blocking:
                self.subscriber.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            
            # Try to decode message
            try:
                decoded = msg.decode('utf-8')
            except UnicodeDecodeError:
                # Skip binary/non-UTF-8 messages (like raw frames)
                return None
            
            # Parse message
            if "|" in decoded:
                topic, data = decoded.split("|", 1)
                return (topic, data)
            else:
                return (decoded, "")
                
        except zmq.Again:
            # Timeout - no message available
            return None
        except Exception as e:
            # Only print non-decode errors
            if "decode" not in str(e).lower():
                print(f"Error receiving message: {e}")
            return None
    
    def receive_all_pending(self) -> List[Tuple[str, str]]:
        """
        Receive all pending messages from the queue.
        
        Returns:
            List of (topic, data) tuples
        """
        messages = []
        while True:
            msg = self.receive_message(blocking=False)
            if msg is None:
                break
            messages.append(msg)
        return messages
    
    def disconnect(self):
        """Disconnect from all services"""
        if self.subscriber:
            self.subscriber.close()
            self.subscriber = None
        self.connected = False
        print("Disconnected from all services")
    
    def cleanup(self):
        """Cleanup resources"""
        self.disconnect()
        self.context.term()
        print("Connection manager cleaned up")
    
    def get_service_info(self) -> Dict[str, Dict]:
        """
        Get information about all configured services.
        
        Returns:
            Dictionary with service information
        """
        info = {}
        for key, service in self.services.items():
            info[key] = {
                "name": service.name,
                "port": service.port,
                "address": service.get_address(),
                "topics": service.topics,
                "required": service.required
            }
        return info
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


if __name__ == "__main__":
    """Example usage of ConnectionManager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Connection Manager Test")
    parser.add_argument("--check", action="store_true",
                       help="Only check services without connecting")
    parser.add_argument("--monitor", type=int, default=30,
                       help="Monitor messages for N seconds (default: 30)")
    args = parser.parse_args()
    
    # Create connection manager
    manager = ConnectionManager()
    
    if args.check:
        # Just check services
        results = manager.check_all_services()
        print(f"\nRequired services running: {manager.are_required_services_running()}")
        manager.cleanup()
    else:
        # Connect and monitor
        print("\n=== Service Information ===")
        for key, info in manager.get_service_info().items():
            print(f"\n{info['name']}:")
            print(f"  Address: {info['address']}")
            print(f"  Topics: {', '.join(info['topics'])}")
            print(f"  Required: {info['required']}")
        
        print("\n=== Connecting to services ===")
        if manager.connect():
            print(f"\n=== Monitoring messages for {args.monitor} seconds ===")
            print("Press Ctrl+C to stop\n")
            
            start_time = time.time()
            message_counts = {}
            
            try:
                while time.time() - start_time < args.monitor:
                    msg = manager.receive_message()
                    if msg:
                        topic, data = msg
                        message_counts[topic] = message_counts.get(topic, 0) + 1
                        
                        # Print summary (truncate long data)
                        data_preview = data[:50] + "..." if len(data) > 50 else data
                        print(f"[{topic}] {data_preview}")
                    else:
                        time.sleep(0.01)  # Small sleep to prevent busy waiting
                
                print(f"\n=== Message Statistics ===")
                for topic, count in message_counts.items():
                    print(f"  {topic}: {count} messages")
                    
            except KeyboardInterrupt:
                print("\nStopping...")
            finally:
                manager.cleanup()
        else:
            print("Failed to connect to services")
            manager.cleanup()
