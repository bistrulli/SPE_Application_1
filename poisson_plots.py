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
                  color='skyblue', edgecolor='black', label='Actual gaps')

    # Overlay theoretical exponential PDF
    x = np.linspace(0, 2, 200)
    theoretical_pdf = lambda_rate * np.exp(-lambda_rate * x)
    axes[0,0].plot(x, theoretical_pdf, 'r-', linewidth=2, label='Expected (exponential)')

    # Add annotation
    axes[0,0].annotate('Most gaps are short!\n(Few long gaps)',
                      xy=(0.8, theoretical_pdf[int(0.8*len(x))]),
                      xytext=(1.2, max(theoretical_pdf)*0.7),
                      arrowprops=dict(arrowstyle='->', color='darkred'),
                      fontsize=10, ha='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    axes[0,0].set_xlabel('Time between consecutive requests (seconds)')
    axes[0,0].set_ylabel('Probability density')
    axes[0,0].set_title('Key Insight: Most requests arrive close together', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Top right: Timeline showing real arrival pattern
    arrival_times = np.cumsum(inter_arrivals[:50])
    axes[0,1].scatter(arrival_times, range(1, 51), alpha=0.7, s=60, color='green')

    # Add average rate line
    axes[0,1].plot([0, arrival_times[-1]], [0, 50], 'r--', linewidth=2,
                   label=f'Average rate: {lambda_rate} req/s')

    # Highlight burst periods
    burst_periods = []
    for i in range(1, len(arrival_times)):
        if arrival_times[i] - arrival_times[i-1] < 0.1:  # Quick succession
            burst_periods.extend([i, i+1])

    if burst_periods:
        axes[0,1].scatter([arrival_times[i-1] for i in set(burst_periods[:10])],
                         [i for i in set(burst_periods[:10])],
                         color='red', s=80, alpha=0.8, label='Bursts')

    axes[0,1].set_xlabel('Time (seconds)')
    axes[0,1].set_ylabel('Request number')
    axes[0,1].set_title('Real Pattern: Requests come in bursts and gaps', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Bottom left: Load variability over time
    window_size = 1.0  # 1 second windows
    max_time = arrival_times[-1]
    time_windows = np.arange(0, max_time, window_size)
    requests_per_window = []

    for t in time_windows:
        count = np.sum((arrival_times >= t) & (arrival_times < t + window_size))
        requests_per_window.append(count)

    axes[1,0].bar(time_windows, requests_per_window, width=window_size*0.8,
                  alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].axhline(lambda_rate * window_size, color='red', linestyle='--',
                     linewidth=2, label=f'Expected: {lambda_rate * window_size:.1f} req/s')

    axes[1,0].set_xlabel('Time window (seconds)')
    axes[1,0].set_ylabel('Requests in window')
    axes[1,0].set_title('SPE Impact: Highly variable load per time window', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Bottom right: Memoryless property - key for modeling
    short_previous = inter_arrivals[1:][inter_arrivals[:-1] < 1/lambda_rate]
    long_previous = inter_arrivals[1:][inter_arrivals[:-1] >= 1/lambda_rate]

    axes[1,1].hist(short_previous, bins=25, density=True, alpha=0.6,
                  color='lightblue', label='After short gap', edgecolor='black')
    axes[1,1].hist(long_previous, bins=25, density=True, alpha=0.6,
                  color='lightcoral', label='After long gap', edgecolor='black')

    # Theoretical curve (should be same for both)
    x = np.linspace(0, 1.5, 100)
    theoretical_pdf = lambda_rate * np.exp(-lambda_rate * x)
    axes[1,1].plot(x, theoretical_pdf, 'k-', linewidth=3, label='Same distribution!')

    axes[1,1].text(0.7, 0.7, 'Memoryless:\nPast doesn\'t\npredict future',
                   transform=axes[1,1].transAxes, fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

    axes[1,1].set_xlabel('Next gap duration (seconds)')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Modeling Advantage: Simple to predict statistically', fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.suptitle('Inter-arrival Times: Why Exponential Distribution Matters for SPE',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Enhanced pedagogical output
    print("Key Insights for Software Performance Engineering:")
    print("=" * 55)

    expected_mean = 1/lambda_rate
    observed_mean = np.mean(inter_arrivals)
    variability = np.std(inter_arrivals) / observed_mean

    print(f"1. TIMING PATTERN:")
    print(f"   • Average gap between requests: {observed_mean:.2f} seconds")
    print(f"   • But gaps are highly variable (CV = {variability:.2f})")
    print(f"   • This creates bursty traffic patterns")

    print(f"\n2. LOAD VARIABILITY:")
    print(f"   • Expected requests per second: {lambda_rate}")
    print(f"   • But actual load varies significantly in short windows")
    print(f"   • Some seconds have many requests, others have none")

    print(f"\n3. MODELING BENEFITS:")
    print(f"   • Memoryless property simplifies analysis")
    print(f"   • Can predict system behavior statistically")
    print(f"   • Enables mathematical performance models (M/M/1)")

    # Statistical validation (simplified)
    ks_stat, ks_pvalue = stats.kstest(inter_arrivals, lambda x: stats.expon.cdf(x, scale=1/lambda_rate))
    print(f"\n4. VALIDATION:")
    print(f"   • Model fits real data: {'✓ YES' if ks_pvalue > 0.05 else '✗ NO'} (p={ks_pvalue:.3f})")

    if len(short_previous) > 10 and len(long_previous) > 10:
        ks_memoryless, p_memoryless = stats.ks_2samp(short_previous, long_previous)
        print(f"   • Memoryless property confirmed: {'✓ YES' if p_memoryless > 0.05 else '✗ NO'}")

    print(f"\n→ Conclusion: Exponential inter-arrivals are realistic for web traffic modeling")


def plot_mm1_utilization_effects(lambda_rate=5, mu_rate=8):
    """
    Visualize the dramatic effect of utilization on M/M/1 system performance
    Focus on the key SPE message: systems become very sensitive near capacity

    Args:
        lambda_rate: Arrival rate (requests/sec)
        mu_rate: Service rate (requests/sec)
    """
    # Calculate current system utilization
    rho = lambda_rate / mu_rate
    E_T = 1 / (mu_rate - lambda_rate)  # Expected response time
    E_N = rho / (1 - rho)              # Expected number in system

    # Generate utilization range (careful near ρ = 1)
    lambda_values = np.linspace(0.1, mu_rate * 0.99, 200)  # Stop before instability
    rho_values = lambda_values / mu_rate
    E_T_values = 1 / (mu_rate - lambda_values)
    E_N_values = rho_values / (1 - rho_values)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top left: Response time vs utilization (key plot!)
    axes[0,0].plot(rho_values, E_T_values, 'b-', linewidth=3, label='Response time')
    axes[0,0].axvline(rho, color='red', linestyle='--', linewidth=2,
                     label=f'Our system (ρ={rho:.2f})')
    axes[0,0].axhline(E_T, color='red', linestyle='--', alpha=0.7, linewidth=2)

    # Highlight danger zone
    danger_zone = rho_values > 0.8
    axes[0,0].fill_between(rho_values[danger_zone], 0, E_T_values[danger_zone],
                          alpha=0.2, color='red', label='Danger zone')

    # Add warning annotation
    axes[0,0].annotate('Response time explodes!\n(System becomes unusable)',
                      xy=(0.9, 10), xytext=(0.6, 8),
                      arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                      fontsize=12, ha='center', fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.8))

    axes[0,0].set_xlabel('Utilization (ρ = λ/μ)', fontsize=12)
    axes[0,0].set_ylabel('Expected Response Time (seconds)', fontsize=12)
    axes[0,0].set_title('Critical SPE Insight: Response Time vs Load', fontsize=14, fontweight='bold')
    axes[0,0].legend(fontsize=11)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0, 15)

    # Top right: Queue length vs utilization
    axes[0,1].plot(rho_values, E_N_values, 'g-', linewidth=3, label='Queue length')
    axes[0,1].axvline(rho, color='red', linestyle='--', linewidth=2,
                     label=f'Our system (ρ={rho:.2f})')
    axes[0,1].axhline(E_N, color='red', linestyle='--', alpha=0.7, linewidth=2)

    # Highlight safe vs danger zones
    safe_zone = rho_values <= 0.8
    axes[0,1].fill_between(rho_values[safe_zone], 0, E_N_values[safe_zone],
                          alpha=0.2, color='green', label='Safe zone')
    danger_zone = rho_values > 0.8
    axes[0,1].fill_between(rho_values[danger_zone], 0, E_N_values[danger_zone],
                          alpha=0.2, color='red', label='Danger zone')

    axes[0,1].set_xlabel('Utilization (ρ = λ/μ)', fontsize=12)
    axes[0,1].set_ylabel('Expected Number in System', fontsize=12)
    axes[0,1].set_title('Queue Builds Up Rapidly Near Capacity', fontsize=14, fontweight='bold')
    axes[0,1].legend(fontsize=11)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, 25)

    # Bottom left: Arrival rate vs System throughput (extended beyond saturation)
    # Extend range beyond saturation to show the concept
    extended_lambda = np.linspace(0.1, mu_rate + 4, 200)

    arrival_rate = extended_lambda  # Arrivals can grow indefinitely
    actual_throughput = np.minimum(extended_lambda, mu_rate)  # System limited by μ

    # Plot arrival rate (continuous line)
    axes[1,0].plot(extended_lambda, arrival_rate, 'b-', linewidth=3, label='Arrival rate')

    # Plot actual system throughput (continuous, plateaus at μ)
    axes[1,0].plot(extended_lambda, actual_throughput, 'red', linewidth=3, label='System throughput')

    # Plot theoretical throughput (dashed line showing what we'd need)
    axes[1,0].plot(extended_lambda, extended_lambda, 'red', linestyle='--', linewidth=2,
                   alpha=0.6, label='Theoretical throughput (needed)')

    # Highlight saturation point
    axes[1,0].axvline(mu_rate, color='black', linestyle='--', linewidth=2,
                     label=f'Saturation point (μ={mu_rate})')

    # Annotate the gap
    axes[1,0].annotate('Gap creates\nqueues and delays',
                      xy=(mu_rate + 2, mu_rate + 1), xytext=(mu_rate + 1, mu_rate + 3),
                      arrowprops=dict(arrowstyle='->', color='darkred'),
                      fontsize=11, ha='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    axes[1,0].set_xlabel('Arrival Rate λ (requests/sec)', fontsize=12)
    axes[1,0].set_ylabel('Rate (requests/sec)', fontsize=12)
    axes[1,0].set_title('System Saturation: Gap Between Arrivals and Throughput', fontsize=14, fontweight='bold')
    axes[1,0].legend(fontsize=10)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xlim(0, mu_rate + 4)
    axes[1,0].set_ylim(0, mu_rate + 4)

    # Bottom right: Capacity planning table
    axes[1,1].axis('off')

    # Create example scenarios
    scenarios = [
        ('Light load', 2, mu_rate, 'green'),
        ('Moderate load', 4, mu_rate, 'orange'),
        ('Heavy load', 6, mu_rate, 'red'),
        ('Critical load', 7.5, mu_rate, 'darkred')
    ]

    table_data = []
    colors = []
    for scenario, lam, mu, color in scenarios:
        rho_s = lam / mu
        E_T_s = 1 / (mu - lam) if lam < mu else float('inf')
        E_N_s = rho_s / (1 - rho_s) if rho_s < 1 else float('inf')
        table_data.append([scenario, f'{lam}', f'{rho_s:.2f}', f'{E_T_s:.2f}s', f'{E_N_s:.1f}'])
        colors.append(color)

    # Create table
    table = axes[1,1].table(cellText=table_data,
                           colLabels=['Scenario', 'λ (req/s)', 'ρ', 'Response Time', 'Queue Length'],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.25, 0.15, 0.15, 0.2, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Color code the rows
    for i, color in enumerate(colors):
        for j in range(5):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)

    axes[1,1].set_title('Capacity Planning: Performance vs Load', fontsize=14, fontweight='bold')

    plt.suptitle('M/M/1 Queue: Why Utilization Management is Critical for SPE',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Key insights output
    print("Critical Performance Engineering Insights:")
    print("=" * 50)

    print(f"CURRENT SYSTEM STATUS:")
    print(f"   • Arrival rate (λ): {lambda_rate} req/s")
    print(f"   • Service rate (μ): {mu_rate} req/s")
    print(f"   • Utilization (ρ): {rho:.1%}")
    print(f"   • Expected response time: {E_T:.2f} seconds")
    print(f"   • Expected queue length: {E_N:.1f} requests")

    print(f"\nCAPACITY PLANNING ASSESSMENT:")
    if rho < 0.7:
        print(f"   Status: SAFE - System has good headroom")
    elif rho < 0.8:
        print(f"   Status: CAUTION - Monitor closely, plan capacity")
    elif rho < 0.9:
        print(f"   Status: WARNING - System under stress, add capacity soon")
    else:
        print(f"   Status: CRITICAL - System near collapse, immediate action needed")

    print(f"\nKEY THEORETICAL INSIGHTS:")
    print(f"   • Small load increases near capacity cause exponential performance degradation")
    print(f"   • Optimal utilization range: 70-80% for stable performance")
    print(f"   • Response time grows non-linearly with utilization")
    print(f"   • System throughput plateaus at service capacity (μ)")

    if rho > 0.8:
        recommended_capacity = lambda_rate / 0.7  # Target 70% utilization
        print(f"\nRECOMMENDATION:")
        print(f"   • Increase service capacity to {recommended_capacity:.1f} req/s")
        print(f"   • This would reduce utilization to 70% for stable performance")