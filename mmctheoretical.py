import math
from typing import Dict


class MMCTheoretical:
    """
    M/M/c theoretical performance calculations.
    
    Multi-server queueing system with:
    - Poisson arrivals (rate λ)
    - Exponential service times (rate μ)
    - c servers
    """
    
    def __init__(self, c: int = 1):
        """
        Initialize M/M/c theoretical calculator.
        
        Args:
            c: Number of servers (default: 1)
        """
        if c < 1:
            raise ValueError("Number of servers must be at least 1")
        self.c = c
    
    def calculate_metrics(self, lambda_rate: float, mu_rate: float, c: int = None) -> Dict[str, float]:
        """
        Calculate M/M/c theoretical metrics at steady state.
        
        Args:
            lambda_rate: Arrival rate (requests/second)
            mu_rate: Service rate per server (requests/second)
            c: Number of servers (if None, uses constructor value)
            
        Returns:
            Dictionary with theoretical metrics:
            - throughput: System throughput (req/s)
            - response_time: Mean time in system (seconds)
            - stable: Whether system is stable
        """
        # Use provided c or fall back to constructor value
        num_servers = c if c is not None else self.c
        
        if num_servers < 1:
            raise ValueError("Number of servers must be at least 1")
        
        # Check stability condition: λ < c*μ
        if lambda_rate >= num_servers * mu_rate:
            return {
                'throughput': float('inf'),
                'response_time': float('inf'),
                'stable': False
            }
        
        # Calculate utilization per server
        rho = lambda_rate / (num_servers * mu_rate)
        
        # Calculate P0 (probability of empty system)
        p0 = self._calculate_p0(lambda_rate, mu_rate, num_servers, rho)
        
        # Calculate Erlang C (probability of queueing)
        erlang_c = self._calculate_erlang_c(lambda_rate, mu_rate, num_servers, rho, p0)
        
        # Calculate mean time in system (response time)
        # W = 1/μ + C(c,a) * ρ / (c * μ * (1 - ρ))
        waiting_time_in_queue = (erlang_c * rho) / (num_servers * mu_rate * (1 - rho))
        service_time = 1 / mu_rate
        response_time = service_time + waiting_time_in_queue
        
        return {
            'throughput': lambda_rate,  # In stable system, throughput = arrival rate
            'response_time': response_time,
            'stable': True
        }
    
    def _calculate_p0(self, lambda_rate: float, mu_rate: float, c: int, rho: float) -> float:
        """
        Calculate P0 (probability that system is empty).
        
        Formula:
        P0 = 1 / [Σ(n=0 to c-1) (a^n / n!) + (a^c / c!) * 1/(1-ρ)]
        where a = λ/μ
        """
        a = lambda_rate / mu_rate
        
        # First sum: Σ(n=0 to c-1) (a^n / n!)
        sum_part = sum(math.pow(a, n) / math.factorial(n) for n in range(c))
        
        # Second part: (a^c / c!) * 1/(1-ρ)
        second_part = (math.pow(a, c) / math.factorial(c)) * (1 / (1 - rho))
        
        p0 = 1 / (sum_part + second_part)
        return p0
    
    def _calculate_erlang_c(self, lambda_rate: float, mu_rate: float, c: int, 
                           rho: float, p0: float) -> float:
        """
        Calculate Erlang C formula (probability of queueing).
        
        Formula:
        C(c,a) = [(a^c / c!) * 1/(1-ρ)] * P0
        where a = λ/μ
        """
        a = lambda_rate / mu_rate
        
        numerator = math.pow(a, c) / math.factorial(c)
        erlang_c = (numerator / (1 - rho)) * p0
        
        return erlang_c