"""
Plotting utilities for Poisson process demonstration
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats


def plot_poisson_distribution(lambda_rate, tau, k_values, probabilities):
    """
    Plot theoretical Poisson distribution and CDF

    Args:
        lambda_rate: Poisson intensity parameter
        tau: Time interval
        k_values: Range of k values
        probabilities: Theoretical probabilities for each k
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Bar plot of probabilities
    axes[0].bar(k_values, probabilities, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(lambda_rate * tau, color='red', linestyle='--',
                    label=f'Expected value: λτ = {lambda_rate * tau}')
    axes[0].set_xlabel('Number of events (k)')
    axes[0].set_ylabel('Probability P(N = k)')
    axes[0].set_title(f'Poisson Distribution: λτ = {lambda_rate * tau}')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Cumulative distribution
    cumulative_prob = np.cumsum(probabilities)
    axes[1].plot(k_values, cumulative_prob, 'o-', color='darkgreen', linewidth=2)
    axes[1].axhline(0.95, color='red', linestyle='--', alpha=0.7, label='95% probability')
    axes[1].set_xlabel('Number of events (k)')
    axes[1].set_ylabel('Cumulative Probability P(N ≤ k)')
    axes[1].set_title('Cumulative Distribution Function')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_simulation_validation(lambda_rate, tau, probabilities, simulated_counts, n_simulations):
    """
    Plot comparison between theoretical and simulated distributions

    Args:
        lambda_rate: Poisson intensity parameter
        tau: Time interval
        probabilities: Theoretical probabilities
        simulated_counts: Array of simulated event counts
        n_simulations: Number of simulations performed
    """
    # Calculate simulated probabilities
    max_k = max(max(simulated_counts), len(probabilities))
    simulated_probs = []
    for k in range(max_k):
        simulated_prob = sum(1 for c in simulated_counts if c == k) / n_simulations
        simulated_probs.append(simulated_prob)

    # Extend theoretical probabilities if needed
    extended_probs = probabilities.copy()
    if len(extended_probs) < max_k:
        for k in range(len(extended_probs), max_k):
            prob = math.exp(-lambda_rate * tau) * (lambda_rate * tau)**k / math.factorial(k)
            extended_probs.append(prob)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Theoretical distribution
    display_range = min(15, len(extended_probs))
    axes[0,0].bar(range(display_range), extended_probs[:display_range], alpha=0.7,
                  color='skyblue', edgecolor='black')
    axes[0,0].axvline(lambda_rate * tau, color='red', linestyle='--',
                    label=f'Expected: λτ = {lambda_rate * tau}')
    axes[0,0].set_xlabel('Number of events (k)')
    axes[0,0].set_ylabel('Probability P(N = k)')
    axes[0,0].set_title('Theoretical Poisson Distribution')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()

    # Simulated distribution
    axes[0,1].bar(range(min(15, len(simulated_probs))), simulated_probs[:15], alpha=0.7,
                  color='lightcoral', edgecolor='black')
    axes[0,1].axvline(np.mean(simulated_counts), color='red', linestyle='--',
                    label=f'Simulated mean: {np.mean(simulated_counts):.2f}')
    axes[0,1].set_xlabel('Number of events (k)')
    axes[0,1].set_ylabel('Probability P(N = k)')
    axes[0,1].set_title(f'Simulated Distribution ({n_simulations:,} samples)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()

    # Side-by-side comparison
    comparison_range = range(12)
    axes[1,0].bar([k-0.2 for k in comparison_range], [extended_probs[k] for k in comparison_range],
                  width=0.4, alpha=0.7, color='skyblue', label='Theoretical', edgecolor='black')
    axes[1,0].bar([k+0.2 for k in comparison_range], [simulated_probs[k] for k in comparison_range],
                  width=0.4, alpha=0.7, color='lightcoral', label='Simulated', edgecolor='black')
    axes[1,0].set_xlabel('Number of events (k)')
    axes[1,0].set_ylabel('Probability P(N = k)')
    axes[1,0].set_title('Theoretical vs Simulated Comparison')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()

    # Difference plot
    differences = [abs(extended_probs[k] - simulated_probs[k]) for k in comparison_range]
    axes[1,1].bar(comparison_range, differences, alpha=0.7, color='orange', edgecolor='black')
    axes[1,1].set_xlabel('Number of events (k)')
    axes[1,1].set_ylabel('|Theoretical - Simulated|')
    axes[1,1].set_title('Absolute Differences')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return simulated_probs, extended_probs


def print_poisson_statistics(lambda_rate, tau, probabilities):
    """Print key statistics for Poisson distribution"""
    print(f"\nKey Statistics:")
    print(f"Sum of probabilities: {sum(probabilities):.6f}")
    print(f"Most likely outcome: {np.argmax(probabilities)} events")
    print(f"P(exactly {lambda_rate * tau} events): {probabilities[lambda_rate * tau]:.4f}")

    cumulative_prob = np.cumsum(probabilities)
    print(f"P(≤ {lambda_rate * tau} events): {cumulative_prob[lambda_rate * tau]:.4f}")

    print(f"\nMost Significant Probabilities:")
    print("-" * 35)
    for k, prob in enumerate(probabilities):
        if prob > 0.01:  # Show only probabilities > 1%
            print(f"P(k = {k:2d}) = {prob:.4f} ({prob*100:.1f}%)")


def print_simulation_comparison(lambda_rate, tau, simulated_counts, simulated_probs, theoretical_probs):
    """Print detailed comparison between simulation and theory"""
    print(f"\nStatistical Summary:")
    print(f"Theoretical mean: {lambda_rate * tau:.3f}")
    print(f"Simulated mean:   {np.mean(simulated_counts):.3f}")
    print(f"Theoretical std:  {np.sqrt(lambda_rate * tau):.3f}")
    print(f"Simulated std:    {np.std(simulated_counts):.3f}")

    print(f"\nDetailed Comparison:")
    print("-" * 55)
    print(f"{'k':<3} {'Theoretical':<12} {'Simulated':<12} {'Difference':<12}")
    print("-" * 55)
    for k in range(12):
        if k < len(theoretical_probs) and theoretical_probs[k] > 0.005:
            diff = abs(theoretical_probs[k] - simulated_probs[k])
            print(f"{k:<3} {theoretical_probs[k]:<12.4f} {simulated_probs[k]:<12.4f} {diff:<12.4f}")


def simple_goodness_of_fit_test(simulated_counts, lambda_rate, tau):
    """
    Simple goodness of fit test comparing means and standard deviations
    More appropriate than KS test for discrete Poisson data

    Args:
        simulated_counts: Array of simulated event counts
        lambda_rate: Poisson intensity parameter
        tau: Time interval

    Returns:
        tuple of (mean_match, std_match, overall_assessment)
    """
    # Theoretical parameters
    theoretical_mean = lambda_rate * tau
    theoretical_std = np.sqrt(lambda_rate * tau)

    # Simulated parameters
    simulated_mean = np.mean(simulated_counts)
    simulated_std = np.std(simulated_counts)

    # Calculate relative errors
    mean_error = abs(simulated_mean - theoretical_mean) / theoretical_mean
    std_error = abs(simulated_std - theoretical_std) / theoretical_std

    # Assess goodness of fit (typical tolerance for large samples)
    mean_match = mean_error < 0.05  # 5% tolerance
    std_match = std_error < 0.10   # 10% tolerance

    print(f"\nGoodness of Fit Assessment:")
    print("-" * 35)
    print(f"Theoretical mean: {theoretical_mean:.3f}")
    print(f"Simulated mean:   {simulated_mean:.3f}")
    print(f"Mean error:       {mean_error:.1%}")
    print(f"Mean match:       {'✓ PASS' if mean_match else '✗ FAIL'}")
    print()
    print(f"Theoretical std:  {theoretical_std:.3f}")
    print(f"Simulated std:    {simulated_std:.3f}")
    print(f"Std error:        {std_error:.1%}")
    print(f"Std match:        {'✓ PASS' if std_match else '✗ FAIL'}")
    print()

    overall_pass = mean_match and std_match
    print(f"Overall result:   {'✓ PASS' if overall_pass else '✗ FAIL'}")

    if overall_pass:
        print("Conclusion: The simulation closely matches the theoretical Poisson distribution")
    else:
        print("Conclusion: Some deviation from theoretical Poisson distribution detected")

    return mean_match, std_match, overall_pass


def two_sample_ks_test(simulated_counts, lambda_rate, tau, n_theoretical_samples=50000):
    """
    Two-sample Kolmogorov-Smirnov test comparing simulated data with
    theoretical samples from Poisson distribution

    Args:
        simulated_counts: Array of simulated event counts
        lambda_rate: Poisson intensity parameter
        tau: Time interval
        n_theoretical_samples: Number of theoretical samples to generate

    Returns:
        ks_statistic, p_value
    """
    from scipy.stats import ks_2samp, poisson

    # Generate theoretical samples from Poisson distribution
    theoretical_samples = poisson.rvs(lambda_rate * tau, size=n_theoretical_samples)

    # Perform two-sample KS test
    ks_stat, p_value = ks_2samp(simulated_counts, theoretical_samples)

    print(f"\nTwo-Sample Kolmogorov-Smirnov Test:")
    print("-" * 40)
    print(f"Simulated samples:   {len(simulated_counts):,}")
    print(f"Theoretical samples: {n_theoretical_samples:,}")
    print(f"KS statistic: {ks_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Result: {'✓ PASS' if p_value > 0.05 else '✗ FAIL'} (α = 0.05)")

    if p_value > 0.05:
        print("Conclusion: No significant difference between simulated and theoretical distributions")
    else:
        print("Conclusion: Significant difference detected between distributions")

    return ks_stat, p_value


def visual_goodness_assessment(simulated_probs, theoretical_probs):
    """
    Visual assessment of goodness of fit by comparing probability distributions

    Args:
        simulated_probs: Simulated probability distribution
        theoretical_probs: Theoretical probability distribution
    """
    print(f"\nVisual Goodness of Fit:")
    print("-" * 25)

    # Calculate maximum absolute difference
    min_len = min(len(simulated_probs), len(theoretical_probs))
    max_diff = max(abs(simulated_probs[k] - theoretical_probs[k]) for k in range(min_len))

    # Calculate total variation distance
    total_variation = 0.5 * sum(abs(simulated_probs[k] - theoretical_probs[k])
                               for k in range(min_len))

    print(f"Maximum absolute difference: {max_diff:.4f}")
    print(f"Total variation distance:    {total_variation:.4f}")

    if max_diff < 0.01 and total_variation < 0.02:
        print("Assessment: Excellent fit")
    elif max_diff < 0.02 and total_variation < 0.05:
        print("Assessment: Good fit")
    elif max_diff < 0.05 and total_variation < 0.10:
        print("Assessment: Acceptable fit")
    else:
        print("Assessment: Poor fit")

    return max_diff, total_variation


def plot_lambda_effect_analysis(lambda_values=[1, 3, 6, 10], time_window=10, n_samples=1000):
    """
    Simple and clear visualization of λ parameter effect on Poisson distributions
    Shows 4 distributions in 2x2 layout

    Args:
        lambda_values: List of lambda values to compare (default: 4 values)
        time_window: Time window for counting events
        n_samples: Number of samples to generate for each lambda
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()  # Make it easier to iterate

    # Colors for different lambda values
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']

    for i, lam in enumerate(lambda_values):
        # Generate samples for this lambda
        samples = [np.random.poisson(lam * time_window) for _ in range(n_samples)]

        # Create histogram
        max_val = max(samples)
        bins = range(max_val + 2)
        axes[i].hist(samples, bins=bins, density=True, alpha=0.7,
                    color=colors[i], edgecolor='black')

        # Add expected value line
        expected = lam * time_window
        axes[i].axvline(expected, color='red', linestyle='--', linewidth=2,
                       label=f'Expected: {expected:.0f}')

        # Add statistics text
        actual_mean = np.mean(samples)
        actual_std = np.std(samples)
        axes[i].text(0.02, 0.95, f'λ = {lam} req/s\nMean: {actual_mean:.1f}\nStd: {actual_std:.1f}',
                    transform=axes[i].transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top')

        # Formatting
        axes[i].set_title(f'Workload Intensity: λ = {lam} req/s', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Number of arrivals in 10 seconds')
        axes[i].set_ylabel('Probability')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        # Set consistent x-axis for comparison
        axes[i].set_xlim(-1, max(lambda_values) * time_window + 40)

    plt.suptitle('Effect of λ on Arrival Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Print comparison summary
    print("Workload Intensity Comparison:")
    print("=" * 45)
    print(f"{'λ (req/s)':<10} {'Expected':<10} {'Actual Mean':<12} {'Std Dev':<10} {'Spread':<10}")
    print("-" * 45)

    for lam in lambda_values:
        samples = [np.random.poisson(lam * time_window) for _ in range(n_samples)]
        expected = lam * time_window
        actual_mean = np.mean(samples)
        actual_std = np.std(samples)
        spread = f"±{actual_std:.1f}"
        print(f"{lam:<10} {expected:<10.0f} {actual_mean:<12.1f} {actual_std:<10.1f} {spread:<10}")

    print(f"\nKey Observations:")
    print(f"• Higher λ → More arrivals on average")
    print(f"• Higher λ → Wider distribution (more variability)")
    print(f"• Higher λ → More intense workload for the system")
    print(f"• The distribution shape remains similar, just shifts and spreads")


def plot_exponential_interarrival_analysis(lambda_rate=5, n_samples=5000):
    """
    Comprehensive analysis of exponential inter-arrival times
    Uses 2x2 layout for detailed visualization

    Args:
        lambda_rate: Poisson process rate parameter
        n_samples: Number of inter-arrival samples to generate
    """
    # Generate inter-arrival times
    inter_arrivals = np.random.exponential(1/lambda_rate, n_samples)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Top left: Histogram with theoretical overlay
    axes[0,0].hist(inter_arrivals, bins=50, density=True, alpha=0.7,
                  color='skyblue', edgecolor='black', label='Simulated')

    # Overlay theoretical exponential PDF
    x = np.linspace(0, 2, 200)
    theoretical_pdf = lambda_rate * np.exp(-lambda_rate * x)
    axes[0,0].plot(x, theoretical_pdf, 'r-', linewidth=2, label='Theoretical Exp(λ)')
    axes[0,0].set_xlabel('Inter-arrival Time (seconds)')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Inter-arrival Times Distribution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Top right: Q-Q plot for goodness of fit
    theoretical_quantiles = stats.expon.ppf(np.linspace(0.01, 0.99, 100), scale=1/lambda_rate)
    sample_quantiles = np.percentile(inter_arrivals, np.linspace(1, 99, 100))

    axes[0,1].scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, s=20)
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='Perfect fit')
    axes[0,1].set_xlabel('Theoretical Quantiles')
    axes[0,1].set_ylabel('Sample Quantiles')
    axes[0,1].set_title('Q-Q Plot: Exponential Fit Assessment')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Bottom left: Arrival timeline
    arrival_times = np.cumsum(inter_arrivals[:50])
    axes[1,0].scatter(arrival_times, range(1, 51), alpha=0.7, s=50, color='green')
    axes[1,0].set_xlabel('Time (seconds)')
    axes[1,0].set_ylabel('Arrival Number')
    axes[1,0].set_title('Timeline of First 50 Arrivals')
    axes[1,0].grid(True, alpha=0.3)

    # Add rate indication
    expected_rate_line = arrival_times[-1] / 50
    axes[1,0].axline((0, 0), slope=1/expected_rate_line, color='red', linestyle='--',
                    label=f'Expected rate: {lambda_rate:.1f} req/s')
    axes[1,0].legend()

    # Bottom right: Memoryless property demonstration
    # Split data into groups based on previous inter-arrival time
    short_previous = inter_arrivals[1:][inter_arrivals[:-1] < 1/lambda_rate]
    long_previous = inter_arrivals[1:][inter_arrivals[:-1] >= 1/lambda_rate]

    axes[1,1].hist(short_previous, bins=30, density=True, alpha=0.6,
                  color='lightblue', label='After short gap', edgecolor='black')
    axes[1,1].hist(long_previous, bins=30, density=True, alpha=0.6,
                  color='lightcoral', label='After long gap', edgecolor='black')

    # Theoretical curve (should be same for both)
    x = np.linspace(0, 1.5, 100)
    theoretical_pdf = lambda_rate * np.exp(-lambda_rate * x)
    axes[1,1].plot(x, theoretical_pdf, 'k-', linewidth=2, label='Theoretical')

    axes[1,1].set_xlabel('Next Inter-arrival Time (seconds)')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Memoryless Property Demonstration')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Statistical validation
    expected_mean = 1/lambda_rate
    observed_mean = np.mean(inter_arrivals)
    expected_std = 1/lambda_rate
    observed_std = np.std(inter_arrivals)

    print("Statistical Validation:")
    print("=" * 30)
    print(f"Expected mean: {expected_mean:.3f}s")
    print(f"Observed mean: {observed_mean:.3f}s")
    print(f"Expected std:  {expected_std:.3f}s")
    print(f"Observed std:  {observed_std:.3f}s")

    # Kolmogorov-Smirnov test (this is appropriate for continuous data)
    ks_stat, ks_pvalue = stats.kstest(inter_arrivals, lambda x: stats.expon.cdf(x, scale=1/lambda_rate))
    print(f"\nGoodness of fit (Kolmogorov-Smirnov test):")
    print(f"KS statistic: {ks_stat:.4f}")
    print(f"p-value: {ks_pvalue:.4f}")
    print(f"Result: {'✓ PASS' if ks_pvalue > 0.05 else '✗ FAIL'} (α=0.05)")

    # Test memoryless property
    if len(short_previous) > 10 and len(long_previous) > 10:
        ks_memoryless, p_memoryless = stats.ks_2samp(short_previous, long_previous)
        print(f"\nMemoryless property test:")
        print(f"KS statistic: {ks_memoryless:.4f}")
        print(f"p-value: {p_memoryless:.4f}")
        print(f"Result: {'✓ PASS' if p_memoryless > 0.05 else '✗ FAIL'} (should be > 0.05)")
        print("Conclusion: Next arrival time is independent of previous gap")