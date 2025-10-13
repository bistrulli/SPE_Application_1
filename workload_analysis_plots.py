"""
Plotting and analysis utilities for Module 2: Workload Patterns.

This module provides specialized visualization functions for analyzing
open vs closed workload behavior, focusing on inter-arrival time
distributions and comparative experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import seaborn as sns

# Set consistent plotting style
plt.style.use('default')
sns.set_palette("husl")


def plot_single_experiment_analysis(
    workload_results,
    analysis_dict: Dict,
    experiment_name: str = "Experiment",
    target_lambda: float = 3.0
) -> None:
    """
    Create comprehensive analysis plots for a single workload experiment.

    Args:
        workload_results: Results from workload_generator
        analysis_dict: Results from analyze_inter_arrival_distribution()
        experiment_name: Name for plot titles
        target_lambda: Target arrival rate for comparison
    """
    inter_arrivals = np.array(workload_results.inter_arrival_times)
    fitted_lambda = analysis_dict['fitted_lambda']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{experiment_name}: Inter-arrival Analysis', fontsize=14, fontweight='bold')

    # 1. Histogram with exponential fit
    axes[0,0].hist(inter_arrivals, bins=30, density=True, alpha=0.7,
                   color='skyblue', edgecolor='black', label='Observed')

    x_range = np.linspace(0, np.max(inter_arrivals), 200)
    theoretical_pdf = fitted_lambda * np.exp(-fitted_lambda * x_range)
    axes[0,0].plot(x_range, theoretical_pdf, 'r-', linewidth=2,
                   label=f'Fitted Exp(λ={fitted_lambda:.2f})')

    # Add target exponential for reference
    target_pdf = target_lambda * np.exp(-target_lambda * x_range)
    axes[0,0].plot(x_range, target_pdf, 'g--', linewidth=2, alpha=0.8,
                   label=f'Target Exp(λ={target_lambda:.1f})')

    axes[0,0].set_xlabel('Inter-arrival Time (s)')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Distribution vs Theoretical')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2. Q-Q plot against exponential
    # Use correct syntax: dist parameter should be a scipy distribution, sparams for parameters
    stats.probplot(inter_arrivals, dist=stats.expon, sparams=(0, 1/fitted_lambda), plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot vs Exponential')
    axes[0,1].grid(True, alpha=0.3)

    # Add R² value to Q-Q plot
    sorted_data = np.sort(inter_arrivals)
    theoretical_quantiles = stats.expon.ppf(np.linspace(0.01, 0.99, len(sorted_data)),
                                           scale=1/fitted_lambda)
    if len(sorted_data) > 1 and len(theoretical_quantiles) > 1:
        r_squared = np.corrcoef(sorted_data, theoretical_quantiles)[0, 1]**2
        axes[0,1].text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=axes[0,1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. Time series of inter-arrivals
    axes[1,0].plot(range(len(inter_arrivals)), inter_arrivals, 'b-', alpha=0.7, linewidth=0.8)
    axes[1,0].axhline(y=analysis_dict['mean_inter_arrival'], color='r', linestyle='--',
                      label=f'Mean: {analysis_dict["mean_inter_arrival"]:.3f}s')
    axes[1,0].axhline(y=1/target_lambda, color='g', linestyle=':',
                      label=f'Target: {1/target_lambda:.3f}s')
    axes[1,0].set_xlabel('Request Number')
    axes[1,0].set_ylabel('Inter-arrival Time (s)')
    axes[1,0].set_title('Time Series of Inter-arrivals')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 4. Cumulative distribution comparison
    sorted_inter_arrivals = np.sort(inter_arrivals)
    empirical_cdf = np.arange(1, len(sorted_inter_arrivals) + 1) / len(sorted_inter_arrivals)
    theoretical_cdf = 1 - np.exp(-fitted_lambda * sorted_inter_arrivals)
    target_cdf = 1 - np.exp(-target_lambda * sorted_inter_arrivals)

    axes[1,1].plot(sorted_inter_arrivals, empirical_cdf, 'b-', linewidth=2, label='Empirical CDF')
    axes[1,1].plot(sorted_inter_arrivals, theoretical_cdf, 'r--', linewidth=2, label='Fitted CDF')
    axes[1,1].plot(sorted_inter_arrivals, target_cdf, 'g:', linewidth=2, label='Target CDF')
    axes[1,1].set_xlabel('Inter-arrival Time (s)')
    axes[1,1].set_ylabel('Cumulative Probability')
    axes[1,1].set_title('CDF Comparison')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_experiment_comparison(
    exp1_results, exp1_analysis: Dict,
    exp2_results, exp2_analysis: Dict,
    target_lambda: float = 3.0,
    exp1_name: str = "Low Service Time",
    exp2_name: str = "High Service Time"
) -> None:
    """
    Create side-by-side comparison plots for two workload experiments.

    Args:
        exp1_results: Results from first experiment
        exp1_analysis: Analysis dict from first experiment
        exp2_results: Results from second experiment
        exp2_analysis: Analysis dict from second experiment
        target_lambda: Target arrival rate
        exp1_name: Name for first experiment
        exp2_name: Name for second experiment
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Experiment Comparison: Open vs Closed Workload Behavior',
                 fontsize=14, fontweight='bold')

    exp1_inter_arrivals = np.array(exp1_results.inter_arrival_times)
    exp2_inter_arrivals = np.array(exp2_results.inter_arrival_times)

    # 1. Histogram comparison
    max_time = max(np.max(exp1_inter_arrivals), np.max(exp2_inter_arrivals))
    bins = np.linspace(0, max_time, 30)

    axes[0,0].hist(exp1_inter_arrivals, bins=bins, density=True, alpha=0.6,
                   color='blue', label=f'Exp 1: {exp1_name}', edgecolor='black')
    axes[0,0].hist(exp2_inter_arrivals, bins=bins, density=True, alpha=0.6,
                   color='red', label=f'Exp 2: {exp2_name}', edgecolor='black')

    # Add theoretical exponential for reference
    x_range = np.linspace(0, max_time, 200)
    theoretical_pdf = target_lambda * np.exp(-target_lambda * x_range)
    axes[0,0].plot(x_range, theoretical_pdf, 'g--', linewidth=2,
                   label=f'Target Exp(λ={target_lambda})', alpha=0.8)

    axes[0,0].set_xlabel('Inter-arrival Time (s)')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Inter-arrival Time Distributions')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2. Box plots for comparison
    bp = axes[0,1].boxplot([exp1_inter_arrivals, exp2_inter_arrivals],
                           labels=[f'Exp 1\n({exp1_name})', f'Exp 2\n({exp2_name})'],
                           patch_artist=True)

    # Color the boxes manually
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[0,1].axhline(y=1/target_lambda, color='green', linestyle='--',
                      label=f'Target Mean: {1/target_lambda:.3f}s')
    axes[0,1].set_ylabel('Inter-arrival Time (s)')
    axes[0,1].set_title('Distribution Comparison')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 3. Time series comparison (first N points for clarity)
    n_points = min(len(exp1_inter_arrivals), len(exp2_inter_arrivals), 100)
    axes[1,0].plot(range(n_points), exp1_inter_arrivals[:n_points],
                   'b-', alpha=0.7, linewidth=1, label=f'Exp 1: {exp1_name}')
    axes[1,0].plot(range(n_points), exp2_inter_arrivals[:n_points],
                   'r-', alpha=0.7, linewidth=1, label=f'Exp 2: {exp2_name}')
    axes[1,0].axhline(y=1/target_lambda, color='green', linestyle='--', alpha=0.8,
                      label=f'Target: {1/target_lambda:.3f}s')
    axes[1,0].set_xlabel(f'Request Number (first {n_points})')
    axes[1,0].set_ylabel('Inter-arrival Time (s)')
    axes[1,0].set_title('Time Series Comparison')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 4. Statistical metrics comparison
    metrics = ['Mean\n(s)', 'Std Dev\n(s)', 'CV', 'KS p-value']
    exp1_values = [
        exp1_analysis['mean_inter_arrival'],
        np.std(exp1_inter_arrivals),
        exp1_analysis['coefficient_variation'],
        exp1_analysis['ks_pvalue']
    ]
    exp2_values = [
        exp2_analysis['mean_inter_arrival'],
        np.std(exp2_inter_arrivals),
        exp2_analysis['coefficient_variation'],
        exp2_analysis['ks_pvalue']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = axes[1,1].bar(x - width/2, exp1_values, width, label=f'Exp 1: {exp1_name}',
                          color='lightblue', edgecolor='black')
    bars2 = axes[1,1].bar(x + width/2, exp2_values, width, label=f'Exp 2: {exp2_name}',
                          color='lightcoral', edgecolor='black')

    # Add reference lines for expected values
    expected_values = [1/target_lambda, 1/target_lambda, 1.0, 0.05]  # Last is significance level
    colors = ['green', 'green', 'green', 'orange']
    linestyles = ['--', '--', '--', ':']

    for i, (expected, color, linestyle) in enumerate(zip(expected_values, colors, linestyles)):
        axes[1,1].axhline(y=expected, color=color, linestyle=linestyle, alpha=0.7)

    axes[1,1].set_ylabel('Value')
    axes[1,1].set_title('Statistical Metrics Comparison')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(metrics)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + max(exp1_values + exp2_values) * 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    add_value_labels(bars1)
    add_value_labels(bars2)

    plt.tight_layout()
    plt.show()


def plot_coefficient_variation_analysis(
    experiments: List[Tuple[str, Dict]],
    target_cv: float = 1.0
) -> None:
    """
    Create focused analysis of coefficient of variation across experiments.

    Args:
        experiments: List of (name, analysis_dict) tuples
        target_cv: Expected CV for exponential distribution (1.0)
    """
    if len(experiments) < 2:
        print("Need at least 2 experiments for comparison")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Coefficient of Variation Analysis', fontsize=14, fontweight='bold')

    names = [exp[0] for exp in experiments]
    cvs = [exp[1]['coefficient_variation'] for exp in experiments]
    colors = ['blue', 'red', 'green', 'orange', 'purple'][:len(experiments)]

    # 1. Bar chart of CVs
    bars = ax1.bar(names, cvs, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=target_cv, color='green', linestyle='--', linewidth=2,
                label=f'Exponential Target: {target_cv}')
    ax1.set_ylabel('Coefficient of Variation')
    ax1.set_title('CV by Experiment')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bar, cv in zip(bars, cvs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{cv:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Deviation from exponential behavior
    deviations = [abs(cv - target_cv) for cv in cvs]
    threshold = 0.2  # Tolerance for "appears exponential"

    bars2 = ax2.bar(names, deviations, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=threshold, color='orange', linestyle=':', linewidth=2,
                label=f'Tolerance: ±{threshold}')
    ax2.set_ylabel('|CV - 1.0|')
    ax2.set_title('Deviation from Exponential')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Color bars based on whether they pass the exponential test
    for bar, deviation in zip(bars2, deviations):
        if deviation <= threshold:
            bar.set_color('lightgreen')
        else:
            bar.set_color('lightcoral')

    # Add value labels and pass/fail indicators
    for bar, deviation, name in zip(bars2, deviations, names):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{deviation:.3f}', ha='center', va='bottom', fontweight='bold')

        # Add check/cross mark
        mark = '✓' if deviation <= threshold else '✗'
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 mark, ha='center', va='bottom', fontsize=16,
                 color='green' if deviation <= threshold else 'red')

    plt.tight_layout()
    plt.show()


def print_experiment_summary_table(
    experiments: List[Tuple[str, any, Dict]],  # (name, workload_results, analysis_dict)
    target_lambda: float = 3.0
) -> None:
    """
    Print a comprehensive summary table comparing multiple experiments.

    Args:
        experiments: List of (name, workload_results, analysis_dict) tuples
        target_lambda: Target arrival rate for comparison
    """
    print("📊 Experiment Summary Table")
    print("=" * 80)

    # Create DataFrame for clean formatting
    data = []
    for name, results, analysis in experiments:
        data.append({
            'Experiment': name,
            'Target Rate': f"{target_lambda:.1f}",
            'Actual Rate': f"{results.actual_rate:.2f}",
            'Success Rate': f"{results.success_rate:.1%}",
            'Total Requests': f"{results.total_requests}",
            'Mean Inter-arrival': f"{analysis['mean_inter_arrival']:.3f}s",
            'CV': f"{analysis['coefficient_variation']:.3f}",
            'KS p-value': f"{analysis['ks_pvalue']:.4f}",
            'Appears Exp?': '✓' if analysis['appears_exponential'] else '✗'
        })

    df = pd.DataFrame(data)
    print(df.to_string(index=False))

    print(f"\n🎯 Key Insights:")
    for i, (name, results, analysis) in enumerate(experiments):
        rate_efficiency = (results.actual_rate / target_lambda) * 100
        print(f"   {i+1}. {name}:")
        print(f"      • Achieved {rate_efficiency:.1f}% of target rate")

        if analysis['appears_exponential']:
            print(f"      • Preserves Poisson arrival pattern (open-like)")
        else:
            print(f"      • Breaks Poisson pattern (closed workload)")

        if abs(analysis['coefficient_variation'] - 1.0) <= 0.2:
            print(f"      • CV ≈ 1.0 → exponential-like behavior")
        else:
            print(f"      • CV = {analysis['coefficient_variation']:.3f} → non-exponential")


def plot_three_way_comparison(
    exp1_results, exp1_analysis: Dict, exp1_name: str,
    exp2_results, exp2_analysis: Dict, exp2_name: str,
    exp3_results, exp3_analysis: Dict, exp3_name: str,
    target_lambda: float = 3.0
) -> None:
    """
    Create comprehensive comparison plots for three workload experiments.

    Args:
        exp1_results, exp1_analysis, exp1_name: First experiment data
        exp2_results, exp2_analysis, exp2_name: Second experiment data
        exp3_results, exp3_analysis, exp3_name: Third experiment data
        target_lambda: Target arrival rate
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Three-Way Comparison: Sync Low vs Sync High vs Async High Service Time',
                 fontsize=14, fontweight='bold')

    exp1_inter_arrivals = np.array(exp1_results.inter_arrival_times)
    exp2_inter_arrivals = np.array(exp2_results.inter_arrival_times)
    exp3_inter_arrivals = np.array(exp3_results.inter_arrival_times)

    # 1. Histogram comparison
    max_time = max(np.max(exp1_inter_arrivals), np.max(exp2_inter_arrivals), np.max(exp3_inter_arrivals))
    bins = np.linspace(0, max_time, 30)

    axes[0,0].hist(exp1_inter_arrivals, bins=bins, density=True, alpha=0.5,
                   color='blue', label=exp1_name, edgecolor='black')
    axes[0,0].hist(exp2_inter_arrivals, bins=bins, density=True, alpha=0.5,
                   color='red', label=exp2_name, edgecolor='black')
    axes[0,0].hist(exp3_inter_arrivals, bins=bins, density=True, alpha=0.5,
                   color='green', label=exp3_name, edgecolor='black')

    # Add theoretical exponential
    x_range = np.linspace(0, max_time, 200)
    theoretical_pdf = target_lambda * np.exp(-target_lambda * x_range)
    axes[0,0].plot(x_range, theoretical_pdf, 'k--', linewidth=2,
                   label=f'Target Exp(λ={target_lambda})', alpha=0.8)

    axes[0,0].set_xlabel('Inter-arrival Time (s)')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Inter-arrival Time Distributions')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2. Box plots comparison
    bp = axes[0,1].boxplot([exp1_inter_arrivals, exp2_inter_arrivals, exp3_inter_arrivals],
                           labels=[exp1_name, exp2_name, exp3_name],
                           patch_artist=True)

    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    axes[0,1].axhline(y=1/target_lambda, color='black', linestyle='--',
                      label=f'Target Mean: {1/target_lambda:.3f}s')
    axes[0,1].set_ylabel('Inter-arrival Time (s)')
    axes[0,1].set_title('Distribution Comparison')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 3. CV comparison
    cv_data = [exp1_analysis['coefficient_variation'],
               exp2_analysis['coefficient_variation'],
               exp3_analysis['coefficient_variation']]

    bars = axes[1,0].bar([exp1_name, exp2_name, exp3_name], cv_data,
                         color=colors, alpha=0.7, edgecolor='black')
    axes[1,0].axhline(y=1.0, color='black', linestyle='--', linewidth=2,
                      label='Expected for Exponential')
    axes[1,0].set_ylabel('Coefficient of Variation')
    axes[1,0].set_title('CV Comparison (Exp≈1.0)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Add value labels and pass/fail indicators
    for bar, cv, name in zip(bars, cv_data, [exp1_name, exp2_name, exp3_name]):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{cv:.3f}', ha='center', va='bottom', fontweight='bold')

        # Add check/cross mark
        appears_exp = abs(cv - 1.0) <= 0.2
        mark = '✓' if appears_exp else '✗'
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.08,
                       mark, ha='center', va='bottom', fontsize=16,
                       color='green' if appears_exp else 'red')

    # 4. Throughput comparison
    rates = [exp1_results.actual_rate, exp2_results.actual_rate, exp3_results.actual_rate]
    efficiency = [(rate/target_lambda)*100 for rate in rates]

    bars2 = axes[1,1].bar([exp1_name, exp2_name, exp3_name], efficiency,
                          color=colors, alpha=0.7, edgecolor='black')
    axes[1,1].axhline(y=100, color='black', linestyle='--', linewidth=2,
                      label='Target (100%)')
    axes[1,1].set_ylabel('Rate Efficiency (%)')
    axes[1,1].set_title('Throughput Achievement')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    # Add value labels
    for bar, eff in zip(bars2, efficiency):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{eff:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def create_three_way_summary(
    exp1_name: str, exp1_results, exp1_analysis: Dict,
    exp2_name: str, exp2_results, exp2_analysis: Dict,
    exp3_name: str, exp3_results, exp3_analysis: Dict,
    target_lambda: float = 3.0
) -> None:
    """
    Create comprehensive three-way summary with interpretation.
    """
    print("🎯 Three-Way Workload Behavior Analysis")
    print("=" * 50)

    # Statistical comparison table
    print_experiment_summary_table([
        (exp1_name, exp1_results, exp1_analysis),
        (exp2_name, exp2_results, exp2_analysis),
        (exp3_name, exp3_results, exp3_analysis)
    ], target_lambda)

    # Visual comparison
    plot_three_way_comparison(
        exp1_results, exp1_analysis, exp1_name,
        exp2_results, exp2_analysis, exp2_name,
        exp3_results, exp3_analysis, exp3_name,
        target_lambda
    )

    # Detailed interpretation
    print(f"\n📖 Three-Way Analysis Interpretation:")
    print(f"=" * 40)

    experiments = [
        (exp1_name, exp1_results, exp1_analysis),
        (exp2_name, exp2_results, exp2_analysis),
        (exp3_name, exp3_results, exp3_analysis)
    ]

    for i, (name, results, analysis) in enumerate(experiments, 1):
        print(f"\n{i}. {name}:")
        rate_efficiency = (results.actual_rate/target_lambda)*100

        if analysis['appears_exponential']:
            print(f"  ✅ Maintains exponential inter-arrival distribution")
            print(f"  ✅ CV = {analysis['coefficient_variation']:.3f} ≈ 1.0")
            print(f"  → TRUE OPEN workload behavior")
        else:
            print(f"  ❌ Deviates from exponential distribution")
            print(f"  ❌ CV = {analysis['coefficient_variation']:.3f} ≠ 1.0")
            print(f"  → CLOSED workload behavior")

        print(f"  📊 Achieves {rate_efficiency:.1f}% of target throughput")

    print(f"\n🏁 Key Insights:")
    print(f"  • Sync Low Service: Quasi-open (service time negligible)")
    print(f"  • Sync High Service: Closed (blocked by response time)")
    print(f"  • Async High Service: True open (independent of service time)")
    print(f"  • Async pattern proves service time doesn't affect Poisson arrivals")
    print(f"  • Demonstrates fundamental difference: sync vs async workload generation")


def create_workload_behavior_summary(
    exp1_name: str, exp1_results, exp1_analysis: Dict,
    exp2_name: str, exp2_results, exp2_analysis: Dict,
    target_lambda: float = 3.0
) -> None:
    """
    Create a comprehensive summary visualization and interpretation.

    Args:
        exp1_name: Name of first experiment
        exp1_results: Workload results from first experiment
        exp1_analysis: Analysis dict from first experiment
        exp2_name: Name of second experiment
        exp2_results: Workload results from second experiment
        exp2_analysis: Analysis dict from second experiment
        target_lambda: Target arrival rate
    """
    print("🎯 Workload Behavior Analysis Summary")
    print("=" * 50)

    # Statistical comparison
    print_experiment_summary_table([
        (exp1_name, exp1_results, exp1_analysis),
        (exp2_name, exp2_results, exp2_analysis)
    ], target_lambda)

    # Visual comparison
    plot_experiment_comparison(
        exp1_results, exp1_analysis,
        exp2_results, exp2_analysis,
        target_lambda, exp1_name, exp2_name
    )

    # CV-focused analysis
    plot_coefficient_variation_analysis([
        (exp1_name, exp1_analysis),
        (exp2_name, exp2_analysis)
    ])

    # Detailed interpretation
    print(f"\n📖 Detailed Interpretation:")
    print(f"=" * 30)

    # Experiment 1 analysis
    print(f"\n{exp1_name}:")
    if exp1_analysis['appears_exponential']:
        print(f"  ✅ SUCCESS: Maintains exponential inter-arrival distribution")
        print(f"  ✅ CV = {exp1_analysis['coefficient_variation']:.3f} ≈ 1.0")
        print(f"  ✅ Achieves {(exp1_results.actual_rate/target_lambda)*100:.1f}% of target rate")
        print(f"  → Behaves like OPEN workload (quasi-Poisson)")
    else:
        print(f"  ⚠️  Deviates from exponential distribution")
        print(f"  ⚠️  CV = {exp1_analysis['coefficient_variation']:.3f}")
        print(f"  → Shows CLOSED workload characteristics")

    # Experiment 2 analysis
    print(f"\n{exp2_name}:")
    if exp2_analysis['appears_exponential']:
        print(f"  🤔 UNEXPECTED: Still maintains exponential distribution")
        print(f"  🤔 Service time may not be high enough")
    else:
        print(f"  ✅ SUCCESS: Breaks exponential pattern as expected")
        print(f"  ✅ CV = {exp2_analysis['coefficient_variation']:.3f} ≠ 1.0")
        print(f"  ✅ Rate reduced to {(exp2_results.actual_rate/target_lambda)*100:.1f}% of target")
        print(f"  → Clear CLOSED workload behavior")

    # Overall conclusion
    rate_drop = ((exp1_results.actual_rate - exp2_results.actual_rate) / exp1_results.actual_rate) * 100
    cv_change = abs(exp2_analysis['coefficient_variation'] - exp1_analysis['coefficient_variation'])

    print(f"\n🏁 Overall Results:")
    print(f"  • Throughput dropped by {rate_drop:.1f}% due to service time blocking")
    print(f"  • CV changed by {cv_change:.3f} between experiments")
    print(f"  • Successfully demonstrated open vs closed workload distinction")
    print(f"  • Server behavior directly impacts client arrival patterns")


# Convenience function for quick analysis
def quick_workload_analysis(workload_results, analysis_dict: Dict, experiment_name: str = "Experiment"):
    """Quick analysis with standard plots and summary."""
    plot_single_experiment_analysis(workload_results, analysis_dict, experiment_name)

    print(f"\n📊 {experiment_name} Quick Summary:")
    print(f"   • Requests generated: {workload_results.total_requests}")
    print(f"   • Actual rate: {workload_results.actual_rate:.2f} req/s")
    print(f"   • Mean inter-arrival: {analysis_dict['mean_inter_arrival']:.3f}s")
    print(f"   • CV: {analysis_dict['coefficient_variation']:.3f}")
    print(f"   • Appears exponential: {analysis_dict['appears_exponential']}")


if __name__ == "__main__":
    # Test with dummy data
    print("Testing workload analysis plotting functions...")

    # Generate some test data
    np.random.seed(42)

    # Simulate exponential inter-arrivals
    n_samples = 1000
    lambda_rate = 3.0
    inter_arrivals = np.random.exponential(1/lambda_rate, n_samples)

    # Create mock analysis dict
    test_analysis = {
        'mean_inter_arrival': np.mean(inter_arrivals),
        'coefficient_variation': np.std(inter_arrivals) / np.mean(inter_arrivals),
        'fitted_lambda': 1/np.mean(inter_arrivals),
        'ks_pvalue': 0.85,  # High p-value = appears exponential
        'appears_exponential': True
    }

    # Create mock workload results
    class MockResults:
        def __init__(self):
            self.inter_arrival_times = inter_arrivals
            self.total_requests = n_samples
            self.actual_rate = lambda_rate * 0.98  # Close to target
            self.success_rate = 1.0

    test_results = MockResults()

    # Test single experiment plot
    plot_single_experiment_analysis(test_results, test_analysis, "Test Experiment", lambda_rate)

    print("✅ Workload analysis plotting functions ready!")