"""
Simple synchronous workload generator for M/M/1 server testing.

This module implements a basic workload generator that:
1. Sleeps for exponentially distributed intervals
2. Makes synchronous HTTP requests to the M/M/1 server
3. Measures actual inter-arrival times between requests

The key insight: with low service times, inter-arrival times should remain
exponential. With high service times, the synchronous nature will "contaminate"
the arrival pattern.
"""

import time
import numpy as np
import requests
import asyncio
import aiohttp
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RequestResult:
    """Result of a single HTTP request."""
    timestamp: float
    response_time: float
    status_code: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class WorkloadResults:
    """Complete results from a workload generation session."""
    lambda_rate: float
    duration: float
    target_url: str
    requests: List[RequestResult]
    inter_arrival_times: List[float]
    actual_rate: float
    success_rate: float

    @property
    def total_requests(self) -> int:
        return len(self.requests)


class SimpleWorkloadGenerator:
    """
    Simple synchronous workload generator.

    Generates requests by:
    1. Sleeping for exponentially distributed time
    2. Making HTTP request (synchronous)
    3. Recording inter-arrival time (time between request completions)
    """

    def __init__(self, target_url: str = "http://localhost:8084", timeout: float = 300.0):
        """
        Initialize the workload generator.

        Args:
            target_url: URL of the M/M/1 server
            timeout: HTTP request timeout in seconds
        """
        self.target_url = target_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def generate_workload(
        self,
        lambda_rate: float,
        duration: float,
        endpoint: str = "/",
        verbose: bool = True
    ) -> WorkloadResults:
        """
        Generate workload for specified duration.

        Args:
            lambda_rate: Target arrival rate (requests per second)
            duration: Duration to run the workload (seconds)
            endpoint: Server endpoint to hit
            verbose: Whether to print progress

        Returns:
            WorkloadResults with timing and response data
        """
        if verbose:
            print(f"Starting workload generation:")
            print(f"  Target rate: {lambda_rate} req/s")
            print(f"  Duration: {duration} seconds")
            print(f"  URL: {self.target_url}{endpoint}")

        requests_data = []
        inter_arrival_times = []
        start_time = time.time()
        last_request_start = None
        request_count = 0

        while time.time() - start_time < duration:
            # 1. Sleep for exponentially distributed time
            sleep_time = np.random.exponential(1.0 / lambda_rate)
            time.sleep(sleep_time)

            # 2. Record request start time
            request_start = time.time()

            # 3. Calculate inter-arrival time (if not first request)
            if last_request_start is not None:
                inter_arrival = request_start - last_request_start
                inter_arrival_times.append(inter_arrival)

            # 4. Make synchronous HTTP request
            try:
                response = self.session.get(
                    f"{self.target_url}{endpoint}",
                    timeout=self.timeout
                )

                request_end = time.time()
                response_time = request_end - request_start

                request_result = RequestResult(
                    timestamp=request_start,
                    response_time=response_time,
                    status_code=response.status_code,
                    success=response.status_code == 200
                )

            except requests.RequestException as e:
                request_end = time.time()
                response_time = request_end - request_start

                request_result = RequestResult(
                    timestamp=request_start,
                    response_time=response_time,
                    status_code=0,
                    success=False,
                    error_message=str(e)
                )

            requests_data.append(request_result)
            last_request_start = request_start  # Track request start times for inter-arrival calculation
            request_count += 1

            if verbose and request_count % 50 == 0:
                elapsed = time.time() - start_time
                current_rate = request_count / elapsed
                print(f"  Sent {request_count} requests, current rate: {current_rate:.2f} req/s")

        # Calculate final statistics
        total_duration = time.time() - start_time
        actual_rate = len(requests_data) / total_duration
        success_count = sum(1 for r in requests_data if r.success)
        success_rate = success_count / len(requests_data) if requests_data else 0

        if verbose:
            print(f"\nWorkload completed:")
            print(f"  Total requests: {len(requests_data)}")
            print(f"  Actual rate: {actual_rate:.2f} req/s")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Inter-arrival times collected: {len(inter_arrival_times)}")

        return WorkloadResults(
            lambda_rate=lambda_rate,
            duration=total_duration,
            target_url=f"{self.target_url}{endpoint}",
            requests=requests_data,
            inter_arrival_times=inter_arrival_times,
            actual_rate=actual_rate,
            success_rate=success_rate
        )

    def close(self):
        """Close the HTTP session."""
        self.session.close()


def analyze_inter_arrival_distribution(results: WorkloadResults) -> dict:
    """
    Analyze the statistical properties of inter-arrival times.

    Args:
        results: WorkloadResults from generate_workload()

    Returns:
        Dictionary with statistical analysis
    """
    if len(results.inter_arrival_times) == 0:
        return {"error": "No inter-arrival times to analyze"}

    inter_arrivals = np.array(results.inter_arrival_times)

    # Basic statistics
    mean_inter_arrival = np.mean(inter_arrivals)
    std_inter_arrival = np.std(inter_arrivals)
    cv = std_inter_arrival / mean_inter_arrival  # Coefficient of variation

    # For exponential distribution, CV should be ~1.0
    theoretical_mean = 1.0 / results.lambda_rate

    # KS test against exponential distribution
    from scipy import stats
    fitted_lambda = 1.0 / mean_inter_arrival
    ks_stat, ks_pvalue = stats.kstest(
        inter_arrivals,
        lambda x: stats.expon.cdf(x, scale=1/fitted_lambda)
    )

    return {
        "sample_size": len(inter_arrivals),
        "mean_inter_arrival": mean_inter_arrival,
        "std_inter_arrival": std_inter_arrival,
        "coefficient_variation": cv,
        "theoretical_mean": theoretical_mean,
        "rate_ratio": mean_inter_arrival / theoretical_mean,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "appears_exponential": ks_pvalue > 0.05 and abs(cv - 1.0) < 0.2,
        "fitted_lambda": fitted_lambda
    }


# Convenience function for quick testing
def quick_test(
    lambda_rate: float = 2.0,
    duration: float = 30.0,
    server_url: str = "http://localhost:8084"
) -> WorkloadResults:
    """
    Quick test function for immediate experimentation.

    Args:
        lambda_rate: Target request rate
        duration: Test duration in seconds
        server_url: M/M/1 server URL

    Returns:
        WorkloadResults
    """
    generator = SimpleWorkloadGenerator(server_url)
    try:
        return generator.generate_workload(lambda_rate, duration)
    finally:
        generator.close()


class AsyncWorkloadGenerator:
    """
    Asynchronous workload generator using asyncio and aiohttp.

    This generator demonstrates TRUE open workload behavior by:
    1. Scheduling requests at exponential intervals (independent of response time)
    2. Not waiting for responses before scheduling the next request
    3. Preserving Poisson arrival pattern regardless of server service time
    """

    def __init__(self, target_url: str = "http://localhost:8084", timeout: float = 300.0):
        """
        Initialize the async workload generator.

        Args:
            target_url: URL of the M/M/1 server
            timeout: HTTP request timeout in seconds
        """
        self.target_url = target_url.rstrip('/')
        self.timeout = timeout

    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str) -> RequestResult:
        """Make a single async HTTP request."""
        request_start = time.time()

        try:
            async with session.get(
                f"{self.target_url}{endpoint}",
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                await response.text()  # Consume response body
                request_end = time.time()

                return RequestResult(
                    timestamp=request_start,
                    response_time=request_end - request_start,
                    status_code=response.status,
                    success=response.status == 200
                )

        except Exception as e:
            request_end = time.time()
            return RequestResult(
                timestamp=request_start,
                response_time=request_end - request_start,
                status_code=0,
                success=False,
                error_message=str(e)
            )

    async def generate_workload(
        self,
        lambda_rate: float,
        duration: float,
        endpoint: str = "/",
        verbose: bool = True
    ) -> WorkloadResults:
        """
        Generate asynchronous workload for specified duration.

        Key difference from sync generator:
        - Requests are scheduled at exponential intervals INDEPENDENTLY
        - Response time does not affect inter-arrival scheduling
        - True Poisson process regardless of server behavior

        Args:
            lambda_rate: Target arrival rate (requests per second)
            duration: Duration to run the workload (seconds)
            endpoint: Server endpoint to hit
            verbose: Whether to print progress

        Returns:
            WorkloadResults with timing and response data
        """
        if verbose:
            print(f"Starting ASYNC workload generation:")
            print(f"  Target rate: {lambda_rate} req/s")
            print(f"  Duration: {duration} seconds")
            print(f"  URL: {self.target_url}{endpoint}")

        requests_data = []
        scheduled_times = []
        start_time = time.time()

        # Pre-generate all request times according to Poisson process
        current_time = 0
        while current_time < duration:
            inter_arrival = np.random.exponential(1.0 / lambda_rate)
            current_time += inter_arrival
            if current_time < duration:
                scheduled_times.append(start_time + current_time)

        if verbose:
            print(f"  Scheduled {len(scheduled_times)} requests")

        # Execute requests asynchronously
        async with aiohttp.ClientSession() as session:
            tasks = []

            for scheduled_time in scheduled_times:
                # Sleep until it's time to send this request
                sleep_time = scheduled_time - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                # Schedule the request (don't wait for completion)
                task = asyncio.create_task(self._make_request(session, endpoint))
                tasks.append(task)

                if verbose and len(tasks) % 50 == 0:
                    elapsed = time.time() - start_time
                    current_rate = len(tasks) / elapsed
                    print(f"  Scheduled {len(tasks)} requests, rate: {current_rate:.2f} req/s")

            # Wait for all requests to complete
            if verbose:
                print("  Waiting for all requests to complete...")

            requests_data = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and convert to proper RequestResult objects
        valid_requests = []
        for result in requests_data:
            if isinstance(result, RequestResult):
                valid_requests.append(result)
            elif isinstance(result, Exception):
                # Create error RequestResult for exceptions
                valid_requests.append(RequestResult(
                    timestamp=time.time(),
                    response_time=0,
                    status_code=0,
                    success=False,
                    error_message=str(result)
                ))

        # Calculate inter-arrival times based on SCHEDULED times (not response times)
        inter_arrival_times = []
        for i in range(1, len(scheduled_times)):
            inter_arrival = scheduled_times[i] - scheduled_times[i-1]
            inter_arrival_times.append(inter_arrival)

        # Calculate statistics
        total_duration = time.time() - start_time
        actual_rate = len(valid_requests) / total_duration
        success_count = sum(1 for r in valid_requests if r.success)
        success_rate = success_count / len(valid_requests) if valid_requests else 0

        if verbose:
            print(f"\nAsync workload completed:")
            print(f"  Total requests: {len(valid_requests)}")
            print(f"  Actual rate: {actual_rate:.2f} req/s")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Inter-arrival times: {len(inter_arrival_times)}")

        return WorkloadResults(
            lambda_rate=lambda_rate,
            duration=total_duration,
            target_url=f"{self.target_url}{endpoint}",
            requests=valid_requests,
            inter_arrival_times=inter_arrival_times,
            actual_rate=actual_rate,
            success_rate=success_rate
        )


# Convenience function for async testing
async def async_quick_test(
    lambda_rate: float = 2.0,
    duration: float = 30.0,
    server_url: str = "http://localhost:8084"
) -> WorkloadResults:
    """
    Quick async test function for immediate experimentation.

    Args:
        lambda_rate: Target request rate
        duration: Test duration in seconds
        server_url: M/M/1 server URL

    Returns:
        WorkloadResults
    """
    generator = AsyncWorkloadGenerator(server_url)
    return await generator.generate_workload(lambda_rate, duration)


def run_async_test(*args, **kwargs):
    """Synchronous wrapper for async_quick_test."""
    return asyncio.run(async_quick_test(*args, **kwargs))


if __name__ == "__main__":
    # Example usage
    print("Testing simple workload generator...")
    results = quick_test(lambda_rate=3.0, duration=20.0)

    print("\nAnalyzing inter-arrival times...")
    analysis = analyze_inter_arrival_distribution(results)

    print(f"Mean inter-arrival: {analysis['mean_inter_arrival']:.3f}s")
    print(f"Coefficient of variation: {analysis['coefficient_variation']:.3f}")
    print(f"Appears exponential: {analysis['appears_exponential']}")
    print(f"KS test p-value: {analysis['ks_pvalue']:.4f}")

    print("\nTesting async workload generator...")
    async_results = run_async_test(lambda_rate=3.0, duration=20.0)
    async_analysis = analyze_inter_arrival_distribution(async_results)

    print(f"Async - Mean inter-arrival: {async_analysis['mean_inter_arrival']:.3f}s")
    print(f"Async - CV: {async_analysis['coefficient_variation']:.3f}")
    print(f"Async - Appears exponential: {async_analysis['appears_exponential']}")