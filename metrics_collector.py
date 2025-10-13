"""
Metrics collector module for gathering data from Prometheus.

This module provides utilities to collect key performance metrics from the
M/M/1 server monitoring stack, including throughput, response times, and
resource utilization.
"""

import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: float
    value: float


@dataclass
class MetricSeries:
    """A time series of metric data."""
    metric_name: str
    labels: Dict[str, str]
    points: List[MetricPoint]

    @property
    def timestamps(self) -> List[float]:
        return [p.timestamp for p in self.points]

    @property
    def values(self) -> List[float]:
        return [p.value for p in self.points]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'value': self.values
        })


class PrometheusCollector:
    """
    Prometheus metrics collector for M/M/1 server monitoring.

    Provides convenient methods to collect key performance metrics during
    workload generation experiments.
    """

    # Pre-defined queries for common M/M/1 metrics
    QUERIES = {
        'throughput': 'rate(envoy_cluster_upstream_rq_total{cluster_name="mm1_service"}[1m])',
        'response_time_avg': 'histogram_quantile(0.5, rate(envoy_cluster_upstream_rq_time_bucket{cluster_name="mm1_service"}[1m]))',
        'response_time_95p': 'histogram_quantile(0.95, rate(envoy_cluster_upstream_rq_time_bucket{cluster_name="mm1_service"}[1m]))',
        'response_time_99p': 'histogram_quantile(0.99, rate(envoy_cluster_upstream_rq_time_bucket{cluster_name="mm1_service"}[1m]))',
        'cpu_usage': 'rate(container_cpu_usage_seconds_total{name="mm1-server"}[1m]) * 100',
        'memory_usage': 'container_memory_usage_bytes{name="mm1-server"}',
        'request_total': 'envoy_cluster_upstream_rq_total{cluster_name="mm1_service"}',
        'success_rate': 'rate(envoy_cluster_upstream_rq_total{cluster_name="mm1_service",envoy_response_code!~"5.."}[1m]) / rate(envoy_cluster_upstream_rq_total{cluster_name="mm1_service"}[1m]) * 100'
    }

    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        """
        Initialize Prometheus collector.

        Args:
            prometheus_url: Base URL of Prometheus server
        """
        self.base_url = prometheus_url.rstrip('/')
        self.session = requests.Session()

    def query(self, query: str, time_point: Optional[float] = None) -> Dict:
        """
        Execute a PromQL query.

        Args:
            query: PromQL query string
            time_point: Unix timestamp for point-in-time query (optional)

        Returns:
            Raw Prometheus API response
        """
        url = f"{self.base_url}/api/v1/query"
        params = {'query': query}
        if time_point:
            params['time'] = time_point

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to query Prometheus: {e}")

    def query_range(
        self,
        query: str,
        start_time: float,
        end_time: float,
        step: str = "5s"
    ) -> Dict:
        """
        Execute a PromQL range query.

        Args:
            query: PromQL query string
            start_time: Start timestamp (Unix time)
            end_time: End timestamp (Unix time)
            step: Query resolution step (e.g., "5s", "1m")

        Returns:
            Raw Prometheus API response
        """
        url = f"{self.base_url}/api/v1/query_range"
        params = {
            'query': query,
            'start': start_time,
            'end': end_time,
            'step': step
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to query Prometheus range: {e}")

    def get_current_metrics(self, metrics: List[str] = None) -> Dict[str, float]:
        """
        Get current values for standard M/M/1 metrics.

        Args:
            metrics: List of metric names to collect (default: all standard metrics)

        Returns:
            Dictionary mapping metric names to current values
        """
        if metrics is None:
            metrics = list(self.QUERIES.keys())

        results = {}
        for metric in metrics:
            if metric not in self.QUERIES:
                raise ValueError(f"Unknown metric: {metric}")

            try:
                response = self.query(self.QUERIES[metric])
                if response['status'] == 'success' and response['data']['result']:
                    # Take the first result if multiple series returned
                    value = float(response['data']['result'][0]['value'][1])
                    results[metric] = value
                else:
                    results[metric] = None
            except Exception as e:
                print(f"Warning: Failed to collect {metric}: {e}")
                results[metric] = None

        return results

    def collect_metrics_during_experiment(
        self,
        duration: float,
        metrics: List[str] = None,
        interval: float = 5.0
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Collect metrics at regular intervals during an experiment.

        Args:
            duration: How long to collect metrics (seconds)
            metrics: Which metrics to collect (default: key performance metrics)
            interval: Collection interval in seconds

        Returns:
            Dictionary mapping metric names to lists of (timestamp, value) tuples
        """
        if metrics is None:
            metrics = ['throughput', 'response_time_avg', 'response_time_95p', 'cpu_usage']

        print(f"Collecting metrics for {duration}s (every {interval}s)...")

        results = {metric: [] for metric in metrics}
        start_time = time.time()
        next_collection = start_time

        while time.time() - start_time < duration:
            current_time = time.time()
            if current_time >= next_collection:
                timestamp = current_time
                current_values = self.get_current_metrics(metrics)

                for metric in metrics:
                    value = current_values.get(metric)
                    if value is not None:
                        results[metric].append((timestamp, value))

                next_collection += interval
                print(f"  Collected at t={current_time-start_time:.1f}s")

            time.sleep(0.5)  # Small sleep to avoid busy waiting

        return results

    def get_metrics_for_timerange(
        self,
        start_time: float,
        end_time: float,
        metrics: List[str] = None,
        step: str = "5s"
    ) -> Dict[str, MetricSeries]:
        """
        Retrieve metrics for a specific time range.

        Args:
            start_time: Start timestamp (Unix time)
            end_time: End timestamp (Unix time)
            metrics: Metrics to retrieve
            step: Query resolution

        Returns:
            Dictionary mapping metric names to MetricSeries objects
        """
        if metrics is None:
            metrics = ['throughput', 'response_time_avg', 'cpu_usage']

        results = {}
        for metric in metrics:
            if metric not in self.QUERIES:
                continue

            try:
                response = self.query_range(
                    self.QUERIES[metric],
                    start_time,
                    end_time,
                    step
                )

                if response['status'] == 'success' and response['data']['result']:
                    # Take first result series
                    series_data = response['data']['result'][0]
                    labels = series_data.get('metric', {})
                    points = []

                    for timestamp, value in series_data['values']:
                        points.append(MetricPoint(
                            timestamp=float(timestamp),
                            value=float(value)
                        ))

                    results[metric] = MetricSeries(
                        metric_name=metric,
                        labels=labels,
                        points=points
                    )

            except Exception as e:
                print(f"Warning: Failed to retrieve {metric}: {e}")

        return results

    def health_check(self) -> bool:
        """
        Check if Prometheus is accessible and responding.

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self.query('up')
            return response['status'] == 'success'
        except:
            return False

    def close(self):
        """Close the HTTP session."""
        self.session.close()


def correlate_workload_and_metrics(
    workload_results,
    metrics_data: Dict[str, List[Tuple[float, float]]],
    time_offset: float = 0.0
) -> pd.DataFrame:
    """
    Correlate workload generator results with collected metrics.

    Args:
        workload_results: Results from workload_generator
        metrics_data: Metrics from collect_metrics_during_experiment
        time_offset: Time offset to align data (seconds)

    Returns:
        DataFrame with aligned workload and server metrics
    """
    # Create base DataFrame from workload results
    workload_df = pd.DataFrame([
        {
            'timestamp': r.timestamp + time_offset,
            'workload_response_time': r.response_time,
            'workload_success': r.success
        }
        for r in workload_results.requests
    ])

    # Add metrics data
    for metric_name, points in metrics_data.items():
        if not points:
            continue

        metric_df = pd.DataFrame(points, columns=['timestamp', metric_name])

        # Merge with workload data using nearest timestamp
        workload_df = pd.merge_asof(
            workload_df.sort_values('timestamp'),
            metric_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )

    return workload_df


# Convenience functions for common use cases
def quick_metrics_snapshot(prometheus_url: str = "http://localhost:9090") -> Dict[str, float]:
    """
    Quick snapshot of current M/M/1 system metrics.

    Args:
        prometheus_url: Prometheus server URL

    Returns:
        Dictionary with current metric values
    """
    collector = PrometheusCollector(prometheus_url)
    try:
        return collector.get_current_metrics()
    finally:
        collector.close()


def monitor_during_workload(
    workload_duration: float,
    prometheus_url: str = "http://localhost:9090"
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Monitor key metrics during a workload generation session.

    Args:
        workload_duration: Duration to monitor (seconds)
        prometheus_url: Prometheus server URL

    Returns:
        Time series data for key metrics
    """
    collector = PrometheusCollector(prometheus_url)
    try:
        return collector.collect_metrics_during_experiment(workload_duration)
    finally:
        collector.close()


if __name__ == "__main__":
    # Example usage
    print("Testing Prometheus collector...")

    collector = PrometheusCollector()

    # Health check
    if not collector.health_check():
        print("❌ Prometheus not accessible")
        exit(1)

    print("✅ Prometheus is healthy")

    # Get current metrics
    metrics = collector.get_current_metrics()
    print("\nCurrent metrics:")
    for name, value in metrics.items():
        if value is not None:
            print(f"  {name}: {value:.3f}")
        else:
            print(f"  {name}: No data")

    collector.close()