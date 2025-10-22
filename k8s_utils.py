"""
Kubernetes utilities for M/M/1 SPE deployment and management.

This module provides helper functions for deploying and managing the M/M/1
system on Kubernetes with Istio service mesh, including service discovery,
configuration updates, and monitoring setup.
"""

import subprocess
import json
import time
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class K8sServiceInfo:
    """Information about a Kubernetes service."""
    name: str
    namespace: str
    ip: str
    ports: List[int]
    ready: bool


class K8sManager:
    """
    Kubernetes management utilities for M/M/1 SPE deployment.

    Provides methods for deploying manifests, service discovery,
    configuration updates, and resource management.
    """

    def __init__(self, namespace: str = "spe-system"):
        """
        Initialize Kubernetes manager.

        Args:
            namespace: Kubernetes namespace for M/M/1 deployment
        """
        self.namespace = namespace
        self.gateway_ip = None
        self.prometheus_port_forward = None
        self.gateway_port_forward = None

    def check_kubectl(self) -> bool:
        """Check if kubectl is available and configured."""
        try:
            result = subprocess.run(
                ['kubectl', 'version', '--client'],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def deploy_manifests(self, manifests_dir: str = "k8s-manifests") -> bool:
        """
        Deploy all Kubernetes manifests.

        Args:
            manifests_dir: Directory containing K8s manifest files

        Returns:
            True if deployment successful, False otherwise
        """
        try:
            print(f"🚀 Deploying K8s manifests from {manifests_dir}/...")

            # Apply all manifests
            result = subprocess.run(
                ['kubectl', 'apply', '-f', f'{manifests_dir}/'],
                capture_output=True, text=True, timeout=60
            )

            if result.returncode != 0:
                print(f"❌ Deployment failed: {result.stderr}")
                return False

            print("✅ Manifests applied successfully")
            print(result.stdout)

            # Wait for rollout
            print("⏳ Waiting for deployment rollout...")
            rollout_result = subprocess.run(
                ['kubectl', 'rollout', 'status', 'deployment/mm1-server',
                 '-n', self.namespace, '--timeout=120s'],
                capture_output=True, text=True, timeout=130
            )

            if rollout_result.returncode == 0:
                print("✅ Deployment rollout completed")
                return True
            else:
                print(f"⚠️ Rollout status check failed: {rollout_result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Deployment timed out")
            return False
        except Exception as e:
            print(f"❌ Deployment error: {e}")
            return False

    def get_pod_status(self) -> Dict:
        """
        Get status of M/M/1 pods.

        Returns:
            Dictionary with pod status information
        """
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-n', self.namespace,
                 '-l', 'app=mm1-server', '-o', 'json'],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                pods = []
                ready_count = 0

                for pod in pods_data['items']:
                    pod_ready = (pod['status']['phase'] == 'Running' and
                               all(condition['status'] == 'True'
                                   for condition in pod['status'].get('conditions', [])
                                   if condition['type'] == 'Ready'))

                    pods.append({
                        'name': pod['metadata']['name'],
                        'status': pod['status']['phase'],
                        'ready': pod_ready
                    })

                    if pod_ready:
                        ready_count += 1

                return {
                    'total_pods': len(pods),
                    'ready_pods': ready_count,
                    'pods': pods
                }
            else:
                return {'error': result.stderr}

        except Exception as e:
            return {'error': str(e)}

    def get_gateway_ip(self) -> str:
        """
        Get Istio Gateway external IP.

        Returns:
            External IP address or 'localhost' if not available
        """
        if self.gateway_ip:
            return self.gateway_ip

        try:
            # Try LoadBalancer IP first
            result = subprocess.run([
                'kubectl', 'get', 'svc', '-n', 'istio-system', 'istio-ingressgateway',
                '-o', 'jsonpath={.status.loadBalancer.ingress[0].ip}'
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and result.stdout.strip():
                self.gateway_ip = result.stdout.strip()
                return self.gateway_ip

            # Fallback to External IP
            result = subprocess.run([
                'kubectl', 'get', 'svc', '-n', 'istio-system', 'istio-ingressgateway',
                '-o', 'jsonpath={.spec.externalIPs[0]}'
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and result.stdout.strip():
                self.gateway_ip = result.stdout.strip()
                return self.gateway_ip

            # Try hostname for cloud providers
            result = subprocess.run([
                'kubectl', 'get', 'svc', '-n', 'istio-system', 'istio-ingressgateway',
                '-o', 'jsonpath={.status.loadBalancer.ingress[0].hostname}'
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and result.stdout.strip():
                self.gateway_ip = result.stdout.strip()
                return self.gateway_ip

            # Last fallback: use localhost with port-forward
            print("⚠️ No external IP found, will use port-forward")
            return "localhost"

        except Exception as e:
            print(f"❌ Failed to get gateway IP: {e}")
            return "localhost"

    def setup_port_forward(self, service: str, local_port: int,
                          service_port: int, namespace: str = None) -> bool:
        """
        Setup port-forward for a service.

        Args:
            service: Service name to port-forward
            local_port: Local port to bind to
            service_port: Service port to forward to
            namespace: Service namespace (defaults to self.namespace)

        Returns:
            True if port-forward successful, False otherwise
        """
        target_namespace = namespace or self.namespace

        try:
            print(f"🔗 Setting up port-forward: localhost:{local_port} -> {service}:{service_port}")

            # Kill any existing port-forward
            subprocess.run(['pkill', '-f', f'port-forward.*{local_port}'],
                         capture_output=True)

            # Start new port-forward in background
            port_forward_proc = subprocess.Popen([
                'kubectl', 'port-forward', '-n', target_namespace,
                f'svc/{service}', f'{local_port}:{service_port}'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Store the process reference
            if service == 'prometheus':
                self.prometheus_port_forward = port_forward_proc
            elif 'gateway' in service.lower():
                self.gateway_port_forward = port_forward_proc

            # Wait for port-forward to establish
            time.sleep(3)

            # Test connection based on service type
            if service == 'prometheus':
                try:
                    response = requests.get(f"http://localhost:{local_port}/api/v1/query",
                                          params={'query': 'up'}, timeout=5)
                    if response.status_code == 200:
                        print("✅ Port-forward established successfully")
                        return True
                except:
                    pass
            else:
                # For other services, just check if port is listening
                import socket
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex(('localhost', local_port))
                    sock.close()
                    if result == 0:
                        print("✅ Port-forward established successfully")
                        return True
                except:
                    pass

            print("❌ Port-forward setup failed")
            return False

        except Exception as e:
            print(f"❌ Port-forward error: {e}")
            return False

    def update_service_time(self, seconds: float) -> bool:
        """
        Update M/M/1 service time via deployment environment variable.

        Args:
            seconds: New service time in seconds

        Returns:
            True if update successful, False otherwise
        """
        try:
            print(f"🔧 Updating service time to {seconds}s...")

            # Update environment variable
            result = subprocess.run([
                'kubectl', 'set', 'env', 'deployment/mm1-server',
                f'SERVICE_TIME_SECONDS={seconds}', '-n', self.namespace
            ], capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                print(f"❌ Failed to update service time: {result.stderr}")
                return False

            # Wait for rollout to complete
            print("⏳ Waiting for rollout to complete...")
            rollout_result = subprocess.run([
                'kubectl', 'rollout', 'status', 'deployment/mm1-server',
                '-n', self.namespace, '--timeout=120s'
            ], capture_output=True, text=True, timeout=130)

            if rollout_result.returncode == 0:
                print(f"✅ Service time updated to {seconds}s")
                time.sleep(5)  # Additional stabilization time
                return True
            else:
                print(f"⚠️ Rollout may have failed: {rollout_result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Service time update failed: {e}")
            return False

    def get_service_info(self, service_name: str, namespace: str = None) -> Optional[K8sServiceInfo]:
        """
        Get information about a Kubernetes service.

        Args:
            service_name: Name of the service
            namespace: Namespace (defaults to self.namespace)

        Returns:
            K8sServiceInfo object or None if not found
        """
        target_namespace = namespace or self.namespace

        try:
            result = subprocess.run([
                'kubectl', 'get', 'svc', service_name, '-n', target_namespace,
                '-o', 'json'
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                svc_data = json.loads(result.stdout)

                # Extract IP
                ip = None
                if svc_data['spec']['type'] == 'LoadBalancer':
                    ingress = svc_data['status'].get('loadBalancer', {}).get('ingress', [])
                    if ingress:
                        ip = ingress[0].get('ip') or ingress[0].get('hostname')

                if not ip:
                    ip = svc_data['spec'].get('clusterIP')

                # Extract ports
                ports = [port['port'] for port in svc_data['spec']['ports']]

                return K8sServiceInfo(
                    name=service_name,
                    namespace=target_namespace,
                    ip=ip or 'unknown',
                    ports=ports,
                    ready=ip is not None
                )
            else:
                return None

        except Exception:
            return None

    def test_service_connectivity(self, url: str, timeout: float = 10) -> Tuple[bool, float, str]:
        """
        Test connectivity to a service URL.

        Args:
            url: Service URL to test
            timeout: Request timeout in seconds

        Returns:
            Tuple of (success, response_time, message)
        """
        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout)
            response_time = time.time() - start_time

            if response.status_code == 200:
                return True, response_time, f"Success ({response.status_code})"
            else:
                return False, response_time, f"HTTP {response.status_code}"

        except requests.RequestException as e:
            return False, 0.0, str(e)

    def cleanup(self):
        """Clean up port-forwards and other resources."""
        if self.prometheus_port_forward:
            self.prometheus_port_forward.terminate()
            self.prometheus_port_forward.wait()
            print("🧹 Prometheus port-forward cleaned up")

        if self.gateway_port_forward:
            self.gateway_port_forward.terminate()
            self.gateway_port_forward.wait()
            print("🧹 Gateway port-forward cleaned up")

    def delete_resources(self, manifests_dir: str = "k8s-manifests") -> bool:
        """
        Delete all Kubernetes resources.

        Args:
            manifests_dir: Directory containing K8s manifest files

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            print(f"🗑️ Deleting K8s resources from {manifests_dir}/...")

            result = subprocess.run(
                ['kubectl', 'delete', '-f', f'{manifests_dir}/'],
                capture_output=True, text=True, timeout=60
            )

            if result.returncode == 0:
                print("✅ Resources deleted successfully")
                return True
            else:
                print(f"⚠️ Some resources may not have been deleted: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Resource deletion failed: {e}")
            return False


# Convenience functions
def quick_deploy(namespace: str = "spe-system", manifests_dir: str = "k8s-manifests") -> K8sManager:
    """
    Quick deployment of M/M/1 system on K8s.

    Args:
        namespace: Kubernetes namespace
        manifests_dir: Directory with manifests

    Returns:
        Configured K8sManager instance
    """
    manager = K8sManager(namespace)

    if not manager.check_kubectl():
        raise RuntimeError("kubectl not available or not configured")

    if manager.deploy_manifests(manifests_dir):
        print("✅ Quick deployment successful")
    else:
        raise RuntimeError("Deployment failed")

    return manager


def setup_monitoring(manager: K8sManager, prometheus_port: int = 9090) -> bool:
    """
    Setup monitoring with port-forward to Prometheus.

    Args:
        manager: K8sManager instance
        prometheus_port: Local port for Prometheus

    Returns:
        True if monitoring setup successful
    """
    return manager.setup_port_forward(
        service='prometheus',
        local_port=prometheus_port,
        service_port=9090,
        namespace='monitoring'
    )


def get_istio_queries() -> Dict[str, str]:
    """
    Get Prometheus queries for Istio metrics.

    Returns:
        Dictionary mapping metric names to PromQL queries
    """
    return {
        'throughput': 'rate(istio_requests_total{destination_service_name="mm1-server",response_code="200"}[1m])',
        'response_time_avg': 'histogram_quantile(0.50, rate(istio_request_duration_milliseconds_bucket{destination_service_name="mm1-server"}[1m])) / 1000',
        'response_time_95p': 'histogram_quantile(0.95, rate(istio_request_duration_milliseconds_bucket{destination_service_name="mm1-server"}[1m])) / 1000',
        'cpu_usage': 'sum(rate(container_cpu_usage_seconds_total{pod=~"mm1-server-.*",cpu="total",container="mm1-server"}[1m]))',
        'memory_usage': '(container_memory_working_set_bytes{pod=~"mm1-server-.*",container="mm1-server"} / container_spec_memory_limit_bytes{pod=~"mm1-server-.*",container="mm1-server"}) * 100',
        'error_rate': 'rate(istio_requests_total{destination_service_name="mm1-server",response_code!~"2.."}[1m]) / rate(istio_requests_total{destination_service_name="mm1-server"}[1m])'
    }


if __name__ == "__main__":
    # Example usage
    print("Testing K8s utilities...")

    manager = K8sManager()

    if manager.check_kubectl():
        print("✅ kubectl available")

        # Check pod status
        status = manager.get_pod_status()
        if 'error' not in status:
            print(f"📊 Pod status: {status['ready_pods']}/{status['total_pods']} ready")

        # Get gateway IP
        gateway_ip = manager.get_gateway_ip()
        print(f"🌐 Gateway IP: {gateway_ip}")

    else:
        print("❌ kubectl not available")