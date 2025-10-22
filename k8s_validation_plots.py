"""
K8s M/M/1 Validation Plotting Module

This module provides specialized plotting functions for M/M/1 validation experiments
running on Kubernetes with Istio service mesh. It handles the comparison between
theoretical predictions and measured cloud-native metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Tuple
import scipy.stats


def plot_k8s_validation_analysis(
    theoretical_predictions: Dict[float, Dict[str, float]],
    measured_metrics: Dict[float, Dict[str, Any]],
    estimated_mu: float
) -> None:
    """
    Create comprehensive M/M/1 validation plots for Kubernetes environment.

    Compares theoretical M/M/1 predictions with measured K8s/Istio metrics across
    different utilization levels.

    Args:
        theoretical_predictions: Dict {lambda_rate: MM1Theoretical.calculate_metrics()}
        measured_metrics: Dict {lambda_rate: measured values from K8s}
        estimated_mu: Estimated service rate from calibration campaign
    """
    if not theoretical_predictions or not measured_metrics:
        print("❌ No validation data to plot")
        return

    # Extract and sort lambda values
    lambda_values = sorted(set(theoretical_predictions.keys()) & set(measured_metrics.keys()))

    if not lambda_values:
        print("❌ No matching lambda values between theoretical and measured data")
        return

    print(f"📊 Plotting K8s validation analysis for {len(lambda_values)} experiments")

    # Prepare data arrays
    theoretical_data = {
        'utilization': [],
        'throughput': [],
        'response_time': [],
        'lambda_rates': []
    }
    measured_data = {
        'utilization': [],
        'throughput': [],
        'response_time': [],
        'response_time_source': []
    }
    utilization_levels = []

    for lambda_rate in lambda_values:
        theory = theoretical_predictions[lambda_rate]
        measured = measured_metrics[lambda_rate]

        # Theoretical values
        theoretical_data['lambda_rates'].append(lambda_rate)
        theoretical_data['utilization'].append(theory['utilization'])
        theoretical_data['throughput'].append(theory['throughput'])
        theoretical_data['response_time'].append(theory['response_time'])

        # Measured values (handle potential None values)
        measured_data['utilization'].append(measured.get('utilization', 0))
        measured_data['throughput'].append(measured.get('throughput', 0))
        measured_data['response_time'].append(measured.get('response_time', 0))
        measured_data['response_time_source'].append(measured.get('response_time_source', 'unknown'))

        utilization_levels.append(theory['utilization'])

    # Create comprehensive validation plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'K8s M/M/1 Model Validation: Theory vs Istio/K8s Measurements\\n'
                f'Estimated Service Rate μ = {estimated_mu:.2f} req/s',
                fontsize=14, fontweight='bold')

    # 1. Throughput validation (theory vs Istio)
    axes[0,0].plot(utilization_levels, theoretical_data['throughput'], 'b-', linewidth=3,
                   marker='o', markersize=10, label='M/M/1 Theory', alpha=0.8)
    axes[0,0].plot(utilization_levels, measured_data['throughput'], 'r--', linewidth=3,
                   marker='s', markersize=10, label='Istio Metrics', alpha=0.8)
    axes[0,0].set_xlabel('Target Utilization (ρ)')
    axes[0,0].set_ylabel('Throughput (req/s)')
    axes[0,0].set_title('Throughput: Theory vs Istio')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xlim(0, max(utilization_levels) * 1.1)

    # 2. Response time validation (theory vs measured)
    axes[0,1].plot(utilization_levels, theoretical_data['response_time'], 'b-', linewidth=3,
                   marker='o', markersize=10, label='M/M/1 Theory', alpha=0.8)

    # Differentiate response time sources in plotting
    client_utils = []
    client_rts = []
    istio_utils = []
    istio_rts = []

    for i, source in enumerate(measured_data['response_time_source']):
        if 'client' in source.lower():
            client_utils.append(utilization_levels[i])
            client_rts.append(measured_data['response_time'][i])
        elif 'istio' in source.lower():
            istio_utils.append(utilization_levels[i])
            istio_rts.append(measured_data['response_time'][i])

    if client_rts:
        axes[0,1].plot(client_utils, client_rts, 'g--', linewidth=3,
                       marker='^', markersize=10, label='Client-side', alpha=0.8)
    if istio_rts:
        axes[0,1].plot(istio_utils, istio_rts, 'r:', linewidth=3,
                       marker='s', markersize=10, label='Istio Proxy', alpha=0.8)

    # If no specific source info, plot all as measured
    if not client_rts and not istio_rts:
        axes[0,1].plot(utilization_levels, measured_data['response_time'], 'r--', linewidth=3,
                       marker='s', markersize=10, label='Measured', alpha=0.8)

    axes[0,1].set_xlabel('Target Utilization (ρ)')
    axes[0,1].set_ylabel('Response Time (s)')
    axes[0,1].set_title('Response Time: Theory vs Measured')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xlim(0, max(utilization_levels) * 1.1)

    # 3. Utilization validation (theory vs K8s CPU)
    axes[1,0].plot(theoretical_data['lambda_rates'], theoretical_data['utilization'], 'b-', linewidth=3,
                   marker='o', markersize=10, label='M/M/1 Theory (λ/μ)', alpha=0.8)
    axes[1,0].plot(theoretical_data['lambda_rates'], measured_data['utilization'], 'orange', linewidth=3,
                   marker='D', markersize=10, label='K8s CPU Usage', linestyle='--', alpha=0.8)
    axes[1,0].set_xlabel('Arrival Rate λ (req/s)')
    axes[1,0].set_ylabel('Utilization')
    axes[1,0].set_title('Utilization: Theory vs K8s CPU')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 4. Correlation scatter plot
    # Combine all metrics for correlation analysis
    all_theoretical = (theoretical_data['throughput'] +
                      theoretical_data['response_time'] +
                      theoretical_data['utilization'])
    all_measured = (measured_data['throughput'] +
                   measured_data['response_time'] +
                   measured_data['utilization'])

    # Filter out zero/invalid values for correlation
    valid_pairs = [(t, m) for t, m in zip(all_theoretical, all_measured)
                   if t > 0 and m > 0 and not np.isnan(t) and not np.isnan(m)]

    if len(valid_pairs) > 3:
        theory_vals, measured_vals = zip(*valid_pairs)

        # Calculate correlation
        correlation, p_value = scipy.stats.pearsonr(theory_vals, measured_vals)

        axes[1,1].scatter(theory_vals, measured_vals, alpha=0.7, s=100,
                         c='purple', label=f'All Metrics\\nr = {correlation:.3f}')

        # Perfect correlation line
        min_val = min(min(theory_vals), min(measured_vals))
        max_val = max(max(theory_vals), max(measured_vals))
        axes[1,1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
                       label='Perfect Correlation')

        axes[1,1].set_xlabel('Theoretical Values')
        axes[1,1].set_ylabel('Measured Values (K8s/Istio)')
        axes[1,1].set_title(f'Theory vs Measurement Correlation\\np-value: {p_value:.4f}')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'Insufficient Data\\nfor Correlation Analysis',
                      ha='center', va='center', transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].set_title('Correlation Analysis')

    plt.tight_layout()
    plt.show()

    print("✅ K8s validation plots generated successfully")


def calculate_k8s_validation_statistics(
    theoretical_predictions: Dict[float, Dict[str, float]],
    measured_metrics: Dict[float, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate statistical validation metrics for K8s M/M/1 experiments.

    Args:
        theoretical_predictions: Theoretical M/M/1 metrics
        measured_metrics: Measured K8s/Istio metrics

    Returns:
        Dictionary with correlation coefficients, errors, and assessment
    """
    # Find common lambda values
    common_lambdas = sorted(set(theoretical_predictions.keys()) & set(measured_metrics.keys()))

    if len(common_lambdas) < 3:
        return {'error': f'Insufficient data points: {len(common_lambdas)}'}

    print(f"📈 Calculating K8s validation statistics for {len(common_lambdas)} experiments")

    # Extract metric arrays
    metrics = ['throughput', 'response_time', 'utilization']
    correlations = {}
    mean_absolute_errors = {}

    for metric in metrics:
        theoretical_vals = []
        measured_vals = []

        for lambda_rate in common_lambdas:
            theory_val = theoretical_predictions[lambda_rate].get(metric)
            measured_val = measured_metrics[lambda_rate].get(metric)

            if theory_val is not None and measured_val is not None and theory_val > 0 and measured_val > 0:
                theoretical_vals.append(theory_val)
                measured_vals.append(measured_val)

        if len(theoretical_vals) >= 3:
            # Calculate Pearson correlation
            correlation, p_value = scipy.stats.pearsonr(theoretical_vals, measured_vals)
            correlations[f'{metric}_corr'] = correlation
            correlations[f'{metric}_p_value'] = p_value

            # Calculate Mean Absolute Relative Error (MARE)
            relative_errors = [abs(measured - theory) / theory * 100
                              for theory, measured in zip(theoretical_vals, measured_vals)]
            mean_absolute_errors[f'{metric}_mare'] = np.mean(relative_errors)

            print(f"  {metric.capitalize()}: r={correlation:.3f} (p={p_value:.4f}), MARE={np.mean(relative_errors):.1f}%")
        else:
            correlations[f'{metric}_corr'] = 0.0
            correlations[f'{metric}_p_value'] = 1.0
            mean_absolute_errors[f'{metric}_mare'] = 100.0
            print(f"  {metric.capitalize()}: insufficient data")

    # Overall assessment
    min_correlation = min([v for k, v in correlations.items() if k.endswith('_corr')])
    max_error = max([v for k, v in mean_absolute_errors.items() if k.endswith('_mare')])

    if min_correlation > 0.8 and max_error < 20:
        assessment = "Excellent model accuracy"
    elif min_correlation > 0.6 and max_error < 30:
        assessment = "Good model accuracy"
    elif min_correlation > 0.4 and max_error < 50:
        assessment = "Moderate model accuracy"
    else:
        assessment = "Poor model accuracy - investigate discrepancies"

    # Response time source analysis
    response_time_sources = {}
    for lambda_rate in common_lambdas:
        source = measured_metrics[lambda_rate].get('response_time_source', 'unknown')
        response_time_sources[source] = response_time_sources.get(source, 0) + 1

    # Compile results
    results = {
        'experiments_count': len(common_lambdas),
        'assessment': assessment,
        'min_correlation': min_correlation,
        'max_error': max_error,
        'response_time_sources': response_time_sources,
        **correlations,
        **{k.replace('mare', 'mae'): v for k, v in mean_absolute_errors.items()}  # Rename for consistency
    }

    print(f"\n📋 Overall Assessment: {assessment}")
    print(f"📊 Min correlation: {min_correlation:.3f}, Max error: {max_error:.1f}%")

    return results


def plot_k8s_experiment_timeline(
    experiment_results: Dict[float, Dict[str, Any]],
    title: str = "K8s M/M/1 Experiment Timeline"
) -> None:
    """
    Plot timeline of K8s validation experiments showing key metrics over time.

    Args:
        experiment_results: Results from K8s validation experiments
        title: Plot title
    """
    if not experiment_results:
        print("❌ No experiment data to plot")
        return

    lambda_rates = sorted(experiment_results.keys())

    # Extract timeline data
    throughputs = [experiment_results[l].get('throughput', 0) for l in lambda_rates]
    response_times = [experiment_results[l].get('response_time', 0) for l in lambda_rates]
    utilizations = [experiment_results[l].get('utilization', 0) for l in lambda_rates]
    success_rates = [experiment_results[l].get('success_rate', 0) for l in lambda_rates]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Throughput over experiments
    axes[0,0].plot(lambda_rates, throughputs, 'b-o', linewidth=2, markersize=8)
    axes[0,0].set_xlabel('Target Arrival Rate λ (req/s)')
    axes[0,0].set_ylabel('Measured Throughput (req/s)')
    axes[0,0].set_title('Throughput vs Target Load')
    axes[0,0].grid(True, alpha=0.3)

    # Response time over experiments
    axes[0,1].plot(lambda_rates, response_times, 'r-s', linewidth=2, markersize=8)
    axes[0,1].set_xlabel('Target Arrival Rate λ (req/s)')
    axes[0,1].set_ylabel('Response Time (s)')
    axes[0,1].set_title('Response Time vs Target Load')
    axes[0,1].grid(True, alpha=0.3)

    # CPU utilization over experiments
    axes[1,0].plot(lambda_rates, [u*100 for u in utilizations], 'g-^', linewidth=2, markersize=8)
    axes[1,0].set_xlabel('Target Arrival Rate λ (req/s)')
    axes[1,0].set_ylabel('CPU Utilization (%)')
    axes[1,0].set_title('K8s CPU Utilization vs Target Load')
    axes[1,0].grid(True, alpha=0.3)

    # Success rate over experiments
    axes[1,1].plot(lambda_rates, [s*100 for s in success_rates], 'm-D', linewidth=2, markersize=8)
    axes[1,1].set_xlabel('Target Arrival Rate λ (req/s)')
    axes[1,1].set_ylabel('Success Rate (%)')
    axes[1,1].set_title('Workload Success Rate vs Target Load')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim(95, 101)  # Focus on success rate variations

    plt.tight_layout()
    plt.show()

    print("✅ K8s experiment timeline plot generated")