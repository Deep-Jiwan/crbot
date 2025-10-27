"""
Health Checker Module for Clash Royale AI Services

This module provides health checking functionality for all bot services including
ZMQ-based services and HTTP endpoints. Can be used standalone or integrated into
other Python applications.
"""

import os
import time
import zmq
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ServiceType(Enum):
    """Types of services that can be checked"""
    ZMQ = "zmq"
    HTTP = "http"
    HTTPS = "https"


@dataclass
class ServiceDefinition:
    """Definition of a service to check"""
    name: str
    type: ServiceType
    port_or_url: str
    required: bool = True
    timeout_ms: int = 1000
    
    def __post_init__(self):
        """Convert port to full URL for ZMQ services"""
        if self.type == ServiceType.ZMQ and isinstance(self.port_or_url, int):
            self.port_or_url = str(self.port_or_url)
    
    def get_address(self) -> str:
        """Get the full address for this service"""
        if self.type == ServiceType.ZMQ:
            if self.port_or_url.startswith("tcp://"):
                return self.port_or_url
            return f"tcp://localhost:{self.port_or_url}"
        else:
            return self.port_or_url


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    service_name: str
    is_healthy: bool
    response_time_ms: float
    error_message: Optional[str] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class HealthChecker:
    """
    Health checker for various service types.
    Supports ZMQ (PUB/SUB) and HTTP/HTTPS endpoints.
    """
    
    def __init__(self):
        """Initialize the health checker"""
        self.services: Dict[str, ServiceDefinition] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self._setup_default_services()
    
    def _setup_default_services(self):
        """Setup default Clash Royale services from environment variables"""
        self.add_service(
            "publisher",
            ServiceType.ZMQ,
            os.getenv("STATE_PUBLISHER_PORT", "5550"),
            required=True
        )
        self.add_service(
            "elixir",
            ServiceType.ZMQ,
            os.getenv("ELIXIR_PORT", "5560"),
            required=True
        )
        self.add_service(
            "cards",
            ServiceType.ZMQ,
            os.getenv("CARDS_PORT", "5590"),
            required=False
        )
        self.add_service(
            "troops",
            ServiceType.ZMQ,
            os.getenv("TROOPS_PORT", "5580"),
            required=False
        )
        self.add_service(
            "win",
            ServiceType.ZMQ,
            os.getenv("WIN_PORT", "5570"),
            required=True
        )
        
        # Add inference service if configured
        inference_url = os.getenv("INFERENCE_LINK")
        if inference_url:
            service_type = ServiceType.HTTPS if inference_url.startswith("https") else ServiceType.HTTP
            self.add_service(
                "inference",
                service_type,
                inference_url,
                required=True,
                timeout_ms=2000
            )
    
    def add_service(self, name: str, service_type: ServiceType, 
                    port_or_url: str, required: bool = True,
                    timeout_ms: int = 1000):
        """
        Add a service to monitor.
        
        Args:
            name: Unique name for the service
            service_type: Type of service (ZMQ, HTTP, HTTPS)
            port_or_url: Port number (for ZMQ) or full URL (for HTTP/HTTPS)
            required: Whether this service is required for operation
            timeout_ms: Timeout for health check in milliseconds
        """
        self.services[name] = ServiceDefinition(
            name=name,
            type=service_type,
            port_or_url=port_or_url,
            required=required,
            timeout_ms=timeout_ms
        )
    
    def remove_service(self, name: str) -> bool:
        """
        Remove a service from monitoring.
        
        Args:
            name: Name of the service to remove
            
        Returns:
            True if service was removed, False if not found
        """
        if name in self.services:
            del self.services[name]
            if name in self.last_results:
                del self.last_results[name]
            return True
        return False
    
    def check_zmq_service(self, service: ServiceDefinition) -> HealthCheckResult:
        """
        Check health of a ZMQ service.
        
        Args:
            service: Service definition to check
            
        Returns:
            HealthCheckResult with status
        """
        start_time = time.time()
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVTIMEO, service.timeout_ms)
        
        try:
            address = service.get_address()
            socket.connect(address)
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            
            # Try to receive a message
            socket.recv()
            
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_name=service.name,
                is_healthy=True,
                response_time_ms=response_time
            )
            
        except zmq.Again:
            # Timeout - service might be running but not publishing
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_name=service.name,
                is_healthy=False,
                response_time_ms=response_time,
                error_message="Timeout: No data received"
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_name=service.name,
                is_healthy=False,
                response_time_ms=response_time,
                error_message=str(e)
            )
            
        finally:
            socket.close()
            context.term()
    
    def check_http_service(self, service: ServiceDefinition) -> HealthCheckResult:
        """
        Check health of an HTTP/HTTPS service.
        
        Args:
            service: Service definition to check
            
        Returns:
            HealthCheckResult with status
        """
        start_time = time.time()
        timeout_sec = service.timeout_ms / 1000.0
        
        try:
            url = service.get_address()
            response = requests.get(url, timeout=timeout_sec)
            response_time = (time.time() - start_time) * 1000
            
            is_healthy = 200 <= response.status_code < 300
            error_msg = None if is_healthy else f"HTTP {response.status_code}"
            
            return HealthCheckResult(
                service_name=service.name,
                is_healthy=is_healthy,
                response_time_ms=response_time,
                error_message=error_msg
            )
            
        except requests.exceptions.Timeout:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_name=service.name,
                is_healthy=False,
                response_time_ms=response_time,
                error_message="Timeout"
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_name=service.name,
                is_healthy=False,
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    def check_service(self, service_name: str) -> Optional[HealthCheckResult]:
        """
        Check health of a specific service.
        
        Args:
            service_name: Name of the service to check
            
        Returns:
            HealthCheckResult or None if service not found
        """
        if service_name not in self.services:
            return None
        
        service = self.services[service_name]
        
        if service.type == ServiceType.ZMQ:
            result = self.check_zmq_service(service)
        else:  # HTTP or HTTPS
            result = self.check_http_service(service)
        
        self.last_results[service_name] = result
        return result
    
    def check_all_services(self, required_only: bool = False,
                          verbose: bool = True) -> Dict[str, HealthCheckResult]:
        """
        Check all registered services.
        
        Args:
            required_only: If True, only check required services
            verbose: If True, print results to console
            
        Returns:
            Dictionary mapping service names to their health check results
        """
        results = {}
        
        if verbose:
            print("\n" + "="*60)
            print("SERVICE HEALTH CHECK")
            print("="*60)
        
        for service_name, service in self.services.items():
            if required_only and not service.required:
                continue
            
            result = self.check_service(service_name)
            if result:
                results[service_name] = result
                
                if verbose:
                    self._print_result(service, result)
        
        if verbose:
            print("="*60)
            all_healthy = self.are_all_required_services_healthy(results)
            status = "✓ ALL REQUIRED SERVICES HEALTHY" if all_healthy else "✗ SOME REQUIRED SERVICES DOWN"
            print(f"{status}\n")
        
        return results
    
    def _print_result(self, service: ServiceDefinition, result: HealthCheckResult):
        """Print a formatted health check result"""
        status = "✓" if result.is_healthy else "✗"
        req = "(required)" if service.required else "(optional)"
        
        # Format address
        if service.type == ServiceType.ZMQ:
            location = f"port {service.port_or_url}"
        else:
            location = service.port_or_url
        
        # Print main status line
        print(f"  {status} {service.name:<12} {location:<30} {req}")
        
        # Print additional info for unhealthy services
        if not result.is_healthy and result.error_message:
            print(f"      └─ Error: {result.error_message}")
        
        # Print response time
        if result.is_healthy:
            print(f"      └─ Response time: {result.response_time_ms:.1f}ms")
    
    def are_all_required_services_healthy(self, 
                                         results: Optional[Dict[str, HealthCheckResult]] = None) -> bool:
        """
        Check if all required services are healthy.
        
        Args:
            results: Optional pre-computed results. If None, will check all services.
            
        Returns:
            True if all required services are healthy, False otherwise
        """
        if results is None:
            results = self.check_all_services(required_only=True, verbose=False)
        
        for service_name, service in self.services.items():
            if service.required:
                result = results.get(service_name)
                if not result or not result.is_healthy:
                    return False
        
        return True
    
    def get_unhealthy_services(self, 
                              results: Optional[Dict[str, HealthCheckResult]] = None) -> List[str]:
        """
        Get list of unhealthy service names.
        
        Args:
            results: Optional pre-computed results. If None, uses last results.
            
        Returns:
            List of service names that are unhealthy
        """
        if results is None:
            results = self.last_results
        
        unhealthy = []
        for service_name, result in results.items():
            if not result.is_healthy:
                unhealthy.append(service_name)
        
        return unhealthy
    
    def get_service_info(self) -> Dict[str, Dict]:
        """
        Get information about all registered services.
        
        Returns:
            Dictionary with service information
        """
        info = {}
        for name, service in self.services.items():
            info[name] = {
                "name": service.name,
                "type": service.type.value,
                "address": service.get_address(),
                "required": service.required,
                "timeout_ms": service.timeout_ms
            }
        return info
    
    def wait_for_services(self, timeout_sec: float = 30, 
                         check_interval: float = 2,
                         required_only: bool = True) -> bool:
        """
        Wait for services to become healthy.
        
        Args:
            timeout_sec: Maximum time to wait in seconds
            check_interval: Time between checks in seconds
            required_only: If True, only wait for required services
            
        Returns:
            True if all services became healthy, False if timeout
        """
        start_time = time.time()
        attempt = 1
        
        print(f"\nWaiting for services to become healthy (timeout: {timeout_sec}s)...")
        
        while time.time() - start_time < timeout_sec:
            results = self.check_all_services(required_only=required_only, verbose=False)
            
            if self.are_all_required_services_healthy(results):
                print(f"✓ All required services are healthy (attempt {attempt})")
                return True
            
            unhealthy = self.get_unhealthy_services(results)
            print(f"  Attempt {attempt}: Waiting for {', '.join(unhealthy)}...")
            
            time.sleep(check_interval)
            attempt += 1
        
        print(f"✗ Timeout waiting for services")
        return False


def main():
    """Main entry point for standalone usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Health Checker for Clash Royale AI Services")
    parser.add_argument("--required-only", action="store_true",
                       help="Check only required services")
    parser.add_argument("--wait", type=float, default=0,
                       help="Wait for services to become healthy (timeout in seconds)")
    parser.add_argument("--continuous", action="store_true",
                       help="Continuously monitor services")
    parser.add_argument("--interval", type=float, default=5,
                       help="Check interval for continuous monitoring (seconds)")
    
    args = parser.parse_args()
    
    # Create health checker
    checker = HealthChecker()
    
    try:
        if args.wait > 0:
            # Wait for services
            success = checker.wait_for_services(
                timeout_sec=args.wait,
                required_only=args.required_only
            )
            exit(0 if success else 1)
            
        elif args.continuous:
            # Continuous monitoring
            print("Starting continuous monitoring (Ctrl+C to stop)...")
            while True:
                checker.check_all_services(required_only=args.required_only)
                time.sleep(args.interval)
                
        else:
            # Single check
            results = checker.check_all_services(required_only=args.required_only)
            all_healthy = checker.are_all_required_services_healthy(results)
            exit(0 if all_healthy else 1)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        exit(0)


if __name__ == "__main__":
    main()
