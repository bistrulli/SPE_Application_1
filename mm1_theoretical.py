"""
M/M/1 theoretical performance calculations for SPE validation experiments.

This module provides the MM1Theoretical class with methods for calculating
theoretical M/M/1 queueing system metrics and estimating service rates from
measured system metrics (both traditional and cloud-native K8s environments).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class MM1Theoretical:
    """
    M/M/1 theoretical performance calculations with service rate estimation.

    This class provides static methods for:
    - Calculating theoretical M/M/1 queueing metrics
    - Estimating service rates from measured utilization and throughput
    - Supporting both traditional and cloud-native (K8s) metric sources
    """

    @staticmethod
    def calculate_metrics(lambda_rate: float, mu_rate: float) -> Dict[str, float]:
        """
        Calculate all M/M/1 theoretical metrics.

        Args:
            lambda_rate: Arrival rate (requests/second)
            mu_rate: Service rate (requests/second)

        Returns:
            Dictionary with theoretical metrics:
            - utilization: System utilization (ρ = λ/μ)
            - response_time: Expected response time E[T] = 1/(μ-λ)
            - system_size: Expected number in system E[N] = ρ/(1-ρ)
            - queue_length: Expected queue length E[Nq] = ρ²/(1-ρ)
            - waiting_time: Expected waiting time E[W] = ρ/(μ(1-ρ))
            - throughput: System throughput (= λ in stable system)
            - stable: Boolean indicating system stability (λ < μ)
        """
        if lambda_rate >= mu_rate:
            return {
                'utilization': float('inf'),
                'response_time': float('inf'),
                'system_size': float('inf'),
                'queue_length': float('inf'),
                'waiting_time': float('inf'),
                'throughput': 0.0,
                'stable': False
            }

        rho = lambda_rate / mu_rate

        return {
            'utilization': rho,
            'response_time': 1 / (mu_rate - lambda_rate),
            'system_size': rho / (1 - rho),
            'queue_length': (rho ** 2) / (1 - rho),
            'waiting_time': rho / (mu_rate * (1 - rho)),
            'throughput': lambda_rate,  # In stable system, throughput = arrival rate
            'stable': True
        }

    @staticmethod
    def estimate_service_rate_from_metrics(utilizations: List[float],
                                         throughputs: List[float]) -> Tuple[Optional[float], Dict]:
        """
        Estimate service rate from measured utilization and throughput metrics.

        Theory: utilization = λ/μ, so μ = λ/utilization ≈ throughput/utilization
        This method works with both traditional system metrics and cloud-native
        K8s metrics (CPU utilization from cAdvisor/K8s native + throughput from Envoy/Istio).

        Args:
            utilizations: List of measured utilization values (0-1 range)
            throughputs: List of measured throughput values (req/s)

        Returns:
            Tuple of (estimated_mu, estimation_stats):
            - estimated_mu: Estimated service rate (req/s) or None if estimation failed
            - estimation_stats: Dictionary with estimation statistics and metadata
        """
        if not utilizations or not throughputs:
            return None, {'error': 'No data provided'}

        # Filter out invalid measurements
        valid_pairs = [(util, tp) for util, tp in zip(utilizations, throughputs)
                      if util > 0.01 and tp > 0.01]  # Avoid division by very small numbers

        if len(valid_pairs) < 5:
            return None, {'error': f'Insufficient valid measurements: {len(valid_pairs)}'}

        # Calculate service rate for each measurement: μ = throughput / utilization
        service_rates = []
        for util, throughput in valid_pairs:
            mu = throughput / util
            service_rates.append(mu)

        # Calculate statistics
        estimated_mu = np.median(service_rates)
        mean_mu = np.mean(service_rates)
        std_mu = np.std(service_rates)
        cv_mu = std_mu / mean_mu if mean_mu > 0 else float('inf')

        estimation_stats = {
            'sample_size': len(service_rates),
            'median_mu': estimated_mu,
            'mean_mu': mean_mu,
            'std_mu': std_mu,
            'cv_mu': cv_mu,
            'min_mu': np.min(service_rates),
            'max_mu': np.max(service_rates),
            'valid_measurements': len(valid_pairs),
            'total_measurements': len(utilizations)
        }

        return estimated_mu, estimation_stats

    @staticmethod
    def estimate_service_rate_from_k8s_metrics(utilizations: List[float],
                                             throughputs: List[float]) -> Tuple[Optional[float], Dict]:
        """
        Estimate service rate from measured K8s utilization and throughput metrics.

        This is an alias for estimate_service_rate_from_metrics() but with explicit
        K8s naming for clarity in cloud-native contexts. The underlying calculation
        is identical: μ = throughput / utilization.

        Theory: utilization = λ/μ, so μ = λ/utilization ≈ throughput/utilization

        Args:
            utilizations: List of measured CPU utilization values from K8s (0-1 range)
            throughputs: List of measured throughput values from Istio/service mesh (req/s)

        Returns:
            Tuple of (estimated_mu, estimation_stats):
            - estimated_mu: Estimated service rate (req/s) or None if estimation failed
            - estimation_stats: Dictionary with estimation statistics and metadata
        """
        return MM1Theoretical.estimate_service_rate_from_metrics(utilizations, throughputs)


if __name__ == "__main__":
    # Example usage and validation
    print("M/M/1 Theoretical Calculator Module")
    print("=" * 35)

    # Test theoretical calculations
    lambda_rate = 3.0  # 3 req/s
    mu_rate = 5.0      # 5 req/s

    metrics = MM1Theoretical.calculate_metrics(lambda_rate, mu_rate)
    print(f"\nExample M/M/1 calculation (λ={lambda_rate}, μ={mu_rate}):")
    for key, value in metrics.items():
        if isinstance(value, float) and value != float('inf'):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Test service rate estimation
    print(f"\nExample service rate estimation:")
    test_utilizations = [0.5, 0.6, 0.55, 0.58, 0.62]
    test_throughputs = [2.5, 3.0, 2.75, 2.9, 3.1]

    estimated_mu, stats = MM1Theoretical.estimate_service_rate_from_metrics(
        test_utilizations, test_throughputs
    )

    if estimated_mu:
        print(f"  Estimated μ: {estimated_mu:.2f} req/s")
        print(f"  Estimation quality: CV = {stats['cv_mu']:.3f}")
    else:
        print(f"  Estimation failed: {stats.get('error', 'Unknown error')}")

    print(f"\n✅ Module validation complete")