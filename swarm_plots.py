import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


def plot_swarm_scaling_results(results_by_replicas: Dict[int, Dict[str, Any]]) -> None:
    """
    Visualize Swarm stress test results across replica counts (client-side metrics only).

    Plots:
    - Throughput vs replicas (client-observed)
    - Response time vs replicas (client-observed)
    """
    if not results_by_replicas:
        print("No Swarm results to plot.")
        return

    # Sort by replica count for consistent plotting
    replica_counts = sorted(results_by_replicas.keys())

    workload_rates = [results_by_replicas[r].get('workload_rate') for r in replica_counts]
    rt_client = [results_by_replicas[r].get('response_time_client') for r in replica_counts]
    success_rates = [results_by_replicas[r].get('success_rate') for r in replica_counts]

    # Convert None to NaN for plotting continuity
    def _nanify(values):
        return [np.nan if v is None else float(v) for v in values]

    workload_rates = _nanify(workload_rates)
    rt_client = _nanify(rt_client)
    success_rates = _nanify(success_rates)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Docker Swarm Scaling Results: mm1-server Replicas (Client-side metrics)', 
                 fontsize=14, fontweight='bold')

    # Throughput plot
    axes[0].plot(replica_counts, workload_rates, marker='o', linewidth=2, markersize=8, 
                 color='blue', label='Client throughput')
    axes[0].set_xlabel('Replicas')
    axes[0].set_ylabel('Throughput (req/s)')
    axes[0].set_title('Throughput vs Replicas')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Response time plot
    axes[1].plot(replica_counts, rt_client, marker='s', linewidth=2, markersize=8,
                 color='red', label='Client response time')
    axes[1].set_xlabel('Replicas')
    axes[1].set_ylabel('Response Time (s)')
    axes[1].set_title('Response Time vs Replicas')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Success rate plot
    axes[2].plot(replica_counts, success_rates, marker='^', linewidth=2, markersize=8,
                 color='green', label='Success rate')
    axes[2].set_xlabel('Replicas')
    axes[2].set_ylabel('Success Rate')
    axes[2].set_title('Success Rate vs Replicas')
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


