"""
K8s Prometheus Metrics Collector - Simple and Working!

This module bypasses the broken PrometheusCollector and makes direct HTTP requests
to Prometheus to collect K8s and Istio metrics for M/M/1 validation.
"""

import requests
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class K8sMetricData:
    """Simple metric data structure for K8s metrics."""
    metric_name: str
    timestamps: List[float]
    values: List[float]

    def __len__(self):
        return len(self.values)

    def mean(self) -> float:
        return np.mean(self.values) if self.values else 0.0

    def sum(self) -> float:
        return np.sum(self.values) if self.values else 0.0


class K8sPrometheusCollector:
    """
    Direct Prometheus collector for K8s/Istio metrics.
    No bullshit, just HTTP requests that work.
    """

    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        """Initialize with Prometheus URL."""
        self.base_url = prometheus_url.rstrip('/')
        self.session = requests.Session()

        # K8s/Istio queries that actually work
        self.queries = {
            'cpu_usage': 'sum(rate(container_cpu_usage_seconds_total{pod=~"mm1-server-.*",cpu="total",container="mm1-server"}[1m]))',
            'throughput': 'sum(rate(istio_requests_total{destination_service_name="mm1-server",response_code="200"}[1m]))',
            'response_time_avg': 'histogram_quantile(0.50, sum(rate(istio_request_duration_milliseconds_bucket{destination_service_name="mm1-server"}[1m])) by (le)) / 1000',
            'error_rate': 'sum(rate(istio_requests_total{destination_service_name="mm1-server",response_code!~"2.."}[1m])) / sum(rate(istio_requests_total{destination_service_name="mm1-server"}[1m]))'
        }

    def health_check(self) -> bool:
        """Check if Prometheus is accessible."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/query",
                                      params={'query': 'up'}, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def query_range(self, query: str, start_time: float, end_time: float, step: str = "15s") -> Optional[K8sMetricData]:
        """
        Execute a range query and return aggregated data.

        Args:
            query: PromQL query
            start_time: Start timestamp (Unix)
            end_time: End timestamp (Unix)
            step: Query step (default: 15s)

        Returns:
            K8sMetricData object with aggregated values or None
        """
        try:
            url = f"{self.base_url}/api/v1/query_range"
            params = {
                'query': query,
                'start': start_time,
                'end': end_time,
                'step': step
            }

            print(f"🔍 Querying: {query[:50]}... from {time.ctime(start_time)} to {time.ctime(end_time)}")

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data['status'] != 'success':
                print(f"❌ Query failed: {data.get('error', 'Unknown error')}")
                return None

            result = data['data']['result']

            if not result:
                print(f"⚠️ No data returned for query")
                return None

            print(f"📊 Found {len(result)} metric series")

            # Aggregate all series (sum across multiple pods/series)
            all_timestamps = []
            all_values = []

            for series in result:
                for timestamp, value in series['values']:
                    try:
                        ts = float(timestamp)
                        val = float(value)
                        if val > 0:  # Only collect non-zero values
                            all_timestamps.append(ts)
                            all_values.append(val)
                    except (ValueError, TypeError):
                        continue

            if not all_values:
                print(f"⚠️ No valid values found in series")
                return None

            print(f"✅ Collected {len(all_values)} data points, range: {np.min(all_values):.4f} - {np.max(all_values):.4f}")

            return K8sMetricData(
                metric_name=query,
                timestamps=all_timestamps,
                values=all_values
            )

        except Exception as e:
            print(f"❌ Query error: {e}")
            return None

    def collect_k8s_metrics(self, start_time: float, end_time: float,
                           metrics: List[str] = None, step: str = "15s") -> Dict[str, K8sMetricData]:
        """
        Collect multiple K8s metrics for a time range.

        Args:
            start_time: Start timestamp (Unix)
            end_time: End timestamp (Unix)
            metrics: List of metric names to collect (default: all)
            step: Query step

        Returns:
            Dictionary of metric_name -> K8sMetricData
        """
        if metrics is None:
            metrics = ['cpu_usage', 'throughput', 'response_time_avg']

        print(f"\n📊 Collecting K8s metrics: {metrics}")
        print(f"📅 Time range: {time.ctime(start_time)} to {time.ctime(end_time)}")

        results = {}

        for metric in metrics:
            if metric not in self.queries:
                print(f"⚠️ Unknown metric: {metric}")
                continue

            query = self.queries[metric]
            data = self.query_range(query, start_time, end_time, step)

            if data:
                results[metric] = data
                print(f"✅ {metric}: {len(data)} points, mean={data.mean():.4f}")
            else:
                print(f"❌ {metric}: No data")

        return results

    def close(self):
        """Close the session."""
        self.session.close()


if __name__ == "__main__":
    # Test the collector
    print("Testing K8s Prometheus Collector")
    print("=" * 35)

    collector = K8sPrometheusCollector()

    if collector.health_check():
        print("✅ Prometheus connection OK")

        # Test with recent data (last 5 minutes)
        end_time = time.time()
        start_time = end_time - 300  # 5 minutes ago

        metrics = collector.collect_k8s_metrics(start_time, end_time)

        print(f"\n📋 Results:")
        for name, data in metrics.items():
            print(f"  {name}: {len(data)} points, mean={data.mean():.4f}")
    else:
        print("❌ Cannot connect to Prometheus")

    collector.close()