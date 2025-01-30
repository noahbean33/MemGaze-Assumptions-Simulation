"""
Multi-Pattern Memory Access Simulator + Advanced Sampling Methods
-----------------------------------------------------------------

1) SAMPLE_PROPORTION is set to ~1% (0.01).
2) After sampling each pattern with all methods, print the method with the highest coverage.

Includes:
- Various memory access patterns (sequential, random, bursty, etc.).
- Multiple sampling techniques (uniform, random, stratified, systematic, weighted, etc.).
- Coverage and reuse interval analysis.
- Subplots comparing how each sampling method captures the pattern.
"""

import random
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter

# ------------------------
#    GLOBAL CONSTANTS
# ------------------------
MEMORY_SIZE = 10_000     # Total number of memory addresses
NUM_ACCESSES = 5_000     # Number of accesses to generate per pattern
SAMPLE_PROPORTION = 0.01 # ~1% of accesses for sampling
BURST_CHANCE = 0.3       # For bursty access

RANDOM_SEED = 42         # For reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create a global memory space
memory_space = list(range(MEMORY_SIZE))

# ------------------------------------------------------
#    MEMORY ACCESS PATTERNS
# ------------------------------------------------------
def sequential_access(mem_space: List[int], num_accesses: int) -> List[int]:
    addresses = []
    for i in range(num_accesses):
        addr = i % len(mem_space)
        addresses.append(mem_space[addr])
    return addresses

def reverse_sequential_access(mem_space: List[int], num_accesses: int) -> List[int]:
    addresses = []
    mem_len = len(mem_space)
    for i in range(num_accesses):
        addr = (mem_len - 1 - i) % mem_len
        addresses.append(mem_space[addr])
    return addresses

def strided_access(mem_space: List[int], num_accesses: int, stride: int = 4) -> List[int]:
    addresses = []
    mem_len = len(mem_space)
    index = 0
    for _ in range(num_accesses):
        addresses.append(mem_space[index])
        index = (index + stride) % mem_len
    return addresses

def random_access(mem_space: List[int], num_accesses: int) -> List[int]:
    mem_len = len(mem_space)
    return [mem_space[random.randint(0, mem_len - 1)] for _ in range(num_accesses)]

def bursty_access(mem_space: List[int],
                  num_accesses: int,
                  burst_chance: float = BURST_CHANCE,
                  burst_size: int = 100,
                  burst_length: int = 10) -> List[int]:
    addresses = []
    mem_len = len(mem_space)
    for _ in range(num_accesses):
        if random.random() < burst_chance:
            base = random.randint(0, max(0, mem_len - burst_size))
            addresses.extend(random.choices(range(base, base + burst_size), k=burst_length))
        else:
            addr = random.randint(0, mem_len - 1)
            addresses.append(addr)
    return addresses[:num_accesses]

def hotspot_access(mem_space: List[int],
                   num_accesses: int,
                   hotspot_size: int = 50,
                   hotspot_probability: float = 0.5) -> List[int]:
    addresses = []
    mem_len = len(mem_space)
    hotspot_start = random.randint(0, max(0, mem_len - hotspot_size))
    hotspot_range = range(hotspot_start, hotspot_start + hotspot_size)
    for _ in range(num_accesses):
        if random.random() < hotspot_probability:
            addresses.append(random.choice(list(hotspot_range)))
        else:
            addresses.append(random.randint(0, mem_len - 1))
    return addresses

def periodic_access(mem_space: List[int], num_accesses: int, period: int = 100) -> List[int]:
    addresses = []
    mem_len = len(mem_space)
    for i in range(num_accesses):
        idx = i % period
        addr = idx % mem_len
        addresses.append(mem_space[addr])
    return addresses

def streaming_access(mem_space: List[int], num_accesses: int) -> List[int]:
    addresses = []
    mem_len = len(mem_space)
    for i in range(num_accesses):
        addr = i % mem_len
        addresses.append(mem_space[addr])
    return addresses

# ----------------------------------------------------------------
#    SAMPLING METHODS (including advanced ones)
# ----------------------------------------------------------------
def uniform_sampling(accesses: List[int], sample_size: int) -> Tuple[List[int], List[int]]:
    """Samples evenly spaced points from the entire sequence."""
    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")
    step = max(1, len(accesses) // sample_size)
    sampled_indices = list(range(0, len(accesses), step))
    sampled_accesses = [accesses[i] for i in sampled_indices]
    return (sampled_indices, sampled_accesses)

def random_sampling(accesses: List[int], sample_proportion: float) -> Tuple[List[int], List[int]]:
    """Randomly samples a fraction of the data."""
    num_samples = int(len(accesses) * sample_proportion)
    sampled_indices = sorted(random.sample(range(len(accesses)), num_samples))
    sampled_accesses = [accesses[i] for i in sampled_indices]
    return (sampled_indices, sampled_accesses)

def stratified_sampling(accesses: List[int], num_bins: int, sample_proportion: float) -> Tuple[List[int], List[int]]:
    """Divides memory addresses into bins based on address % num_bins and samples proportionally from each bin."""
    bins = [[] for _ in range(num_bins)]
    for i, addr in enumerate(accesses):
        bin_index = addr % num_bins
        bins[bin_index].append((i, addr))

    sampled_indices = []
    sampled_accesses = []
    for b in bins:
        sample_size = int(len(b) * sample_proportion)
        if len(b) == 0:
            continue
        chosen = random.sample(b, min(sample_size, len(b)))
        for (idx, val) in chosen:
            sampled_indices.append(idx)
            sampled_accesses.append(val)

    # Sort indices to keep consistent order
    sorted_pairs = sorted(zip(sampled_indices, sampled_accesses), key=lambda x: x[0])
    sampled_indices = [p[0] for p in sorted_pairs]
    sampled_accesses = [p[1] for p in sorted_pairs]
    return (sampled_indices, sampled_accesses)

def systematic_sampling(accesses: List[int], step: int) -> Tuple[List[int], List[int]]:
    """Selects every 'step'-th element starting from a random offset in [0..step-1]."""
    start = random.randint(0, max(0, step - 1))
    sampled_indices = list(range(start, len(accesses), step))
    sampled_accesses = [accesses[i] for i in sampled_indices]
    return (sampled_indices, sampled_accesses)

def weighted_sampling(accesses: List[int], num_samples: int) -> Tuple[List[int], List[int]]:
    """
    Weighted random sampling based on address frequency:
      - Build a distribution over unique addresses (freq-based).
      - Sample addresses with replacement using that distribution.
      - For each chosen address, pick a random index from its occurrences.
    """
    freq = Counter(accesses)
    unique_addrs = list(freq.keys())

    sum_of_freq = sum(freq.values())  # equals len(accesses)
    # Probability distribution over unique addresses
    probabilities = [freq[a] / sum_of_freq for a in unique_addrs]

    # Sample addresses with replacement
    chosen_unique_addrs = np.random.choice(
        unique_addrs,
        size=num_samples,
        replace=True,
        p=probabilities
    )

    # Map address -> list of indices
    index_map = {}
    for i, addr in enumerate(accesses):
        index_map.setdefault(addr, []).append(i)

    # For each chosen address, pick a random index from its occurrences
    chosen_indices = [random.choice(index_map[addr]) for addr in chosen_unique_addrs]
    chosen_indices.sort()
    sampled_accesses = [accesses[i] for i in chosen_indices]

    return (chosen_indices, sampled_accesses)

def cluster_sampling(accesses: List[int], cluster_size: int, num_clusters: int) -> Tuple[List[int], List[int]]:
    """Divides the access list into clusters of size cluster_size, then samples entire clusters."""
    clusters = []
    for start in range(0, len(accesses), cluster_size):
        cluster_indices = list(range(start, min(start+cluster_size, len(accesses))))
        clusters.append(cluster_indices)

    selected_clusters = random.sample(clusters, min(num_clusters, len(clusters)))
    sampled_indices = [idx for cluster in selected_clusters for idx in cluster]
    sampled_indices = sorted(sampled_indices)
    sampled_accesses = [accesses[i] for i in sampled_indices]
    return (sampled_indices, sampled_accesses)

def temporal_sampling(accesses: List[int], window_size: int, sample_proportion: float) -> Tuple[List[int], List[int]]:
    """Splits the access list into windows of size window_size, samples sample_proportion from each window."""
    sampled_indices = []
    for start in range(0, len(accesses), window_size):
        window_indices = list(range(start, min(start+window_size, len(accesses))))
        if not window_indices:
            continue
        sub_size = int(len(window_indices) * sample_proportion)
        if sub_size > 0:
            chosen = random.sample(window_indices, sub_size)
            sampled_indices.extend(chosen)

    sampled_indices = sorted(sampled_indices)
    sampled_accesses = [accesses[i] for i in sampled_indices]
    return (sampled_indices, sampled_accesses)

def adaptive_sampling(accesses: List[int], threshold: int, high_rate: float, low_rate: float) -> Tuple[List[int], List[int]]:
    """
    Example of adaptive sampling:
    If addr > threshold => sample with probability high_rate
    Otherwise => sample with probability low_rate
    """
    sampled_indices = []
    for i, addr in enumerate(accesses):
        if addr > threshold:
            if random.random() < high_rate:
                sampled_indices.append(i)
        else:
            if random.random() < low_rate:
                sampled_indices.append(i)

    sampled_indices.sort()
    sampled_accesses = [accesses[i] for i in sampled_indices]
    return (sampled_indices, sampled_accesses)

def reservoir_sampling(accesses: List[int], reservoir_size: int) -> Tuple[List[int], List[int]]:
    """
    Maintains a reservoir of fixed size. For the i-th incoming element,
    replace a random element in the reservoir with probability 1/i.
    """
    if reservoir_size <= 0:
        raise ValueError("Reservoir size must be positive.")

    n = len(accesses)
    # Initialize reservoir with the first reservoir_size addresses
    reservoir = accesses[:min(reservoir_size, n)]
    # Track which indices are in the reservoir
    sampled_indices = list(range(min(reservoir_size, n)))

    for i in range(reservoir_size, n):
        j = random.randint(0, i)
        if j < reservoir_size:
            reservoir[j] = accesses[i]
            sampled_indices[j] = i

    # Sort final indices
    sorted_pairs = sorted(zip(sampled_indices, reservoir), key=lambda x: x[0])
    final_indices = [p[0] for p in sorted_pairs]
    final_accesses = [p[1] for p in sorted_pairs]
    return (final_indices, final_accesses)

def frequency_based_sampling(accesses: List[int], sample_proportion: float) -> Tuple[List[int], List[int]]:
    """
    Samples addresses based on frequency:
       1) Sort addresses by frequency (descending).
       2) Take the top sample_proportion of them.
       3) Return all indices that match those addresses.
    """
    freq = Counter(accesses)
    sorted_addrs = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)
    cutoff = int(len(sorted_addrs) * sample_proportion)
    chosen_addrs = set(sorted_addrs[:cutoff])

    sampled_indices = [i for i, addr in enumerate(accesses) if addr in chosen_addrs]
    sampled_accesses = [accesses[i] for i in sampled_indices]
    return (sampled_indices, sampled_accesses)

def spatial_sampling(accesses: List[int], num_regions: int, sample_proportion: float) -> Tuple[List[int], List[int]]:
    """
    Divide the address space into 'num_regions' logical regions,
    sample sample_proportion from each region's indices in 'accesses'.
    """
    region_size = max(1, MEMORY_SIZE // num_regions)
    region_buckets = [[] for _ in range(num_regions)]

    for i, addr in enumerate(accesses):
        region_index = min(addr // region_size, num_regions - 1)
        region_buckets[region_index].append(i)

    sampled_indices = []
    for bucket in region_buckets:
        sub_size = int(len(bucket) * sample_proportion)
        if sub_size > 0 and len(bucket) > 0:
            chosen = random.sample(bucket, sub_size)
            sampled_indices.extend(chosen)

    sampled_indices.sort()
    sampled_accesses = [accesses[i] for i in sampled_indices]
    return (sampled_indices, sampled_accesses)

def hybrid_sampling(accesses: List[int]) -> Tuple[List[int], List[int]]:
    """
    Combine random sampling (50%) and stratified sampling (50%).
    """
    total_samples = int(len(accesses) * SAMPLE_PROPORTION)
    half_samples = total_samples // 2

    # Random half
    random_indices = sorted(random.sample(range(len(accesses)), half_samples))

    # Stratified half (use a small number of bins, e.g., 10)
    num_bins = 10
    bins = [[] for _ in range(num_bins)]
    for i, addr in enumerate(accesses):
        bin_index = addr % num_bins
        bins[bin_index].append((i, addr))
    # Proportion for stratified part
    stratified_prop = half_samples / len(accesses)

    strat_indices = []
    for b in bins:
        sample_size = int(len(b) * stratified_prop)
        if sample_size > 0 and len(b) > 0:
            chosen = random.sample(b, sample_size)
            for (idx, _) in chosen:
                strat_indices.append(idx)

    combined_indices = sorted(set(random_indices + strat_indices))
    sampled_accesses = [accesses[i] for i in combined_indices]

    return (combined_indices, sampled_accesses)

# ---------------------------------------------------------------
#      ANALYSIS / HELPER FUNCTIONS
# ---------------------------------------------------------------
def calculate_coverage(accesses: List[int], samples: List[int]) -> float:
    """
    Coverage = (unique addresses in sample) / (unique addresses in full set) * 100
    """
    unique_accesses = set(accesses)
    unique_samples = set(samples)
    if not unique_accesses:
        return 0.0
    return (len(unique_samples) / len(unique_accesses)) * 100

def analyze_temporal_relationships(accesses: List[int], samples: List[int]) -> Dict[int, List[int]]:
    """
    For each address in 'samples', collect the intervals between subsequent appearances.
    """
    reuse_intervals = {}
    last_seen = {}
    sample_set = set(samples)
    for i, addr in enumerate(accesses):
        if addr in sample_set:
            if addr in last_seen:
                interval = i - last_seen[addr]
                reuse_intervals.setdefault(addr, []).append(interval)
            last_seen[addr] = i
    return reuse_intervals

# ---------------------------------------------------------------
#       COMPARISON / VISUALIZATION FOR MULTIPLE SAMPLING
# ---------------------------------------------------------------
def compare_all_sampling_methods_for_pattern(accesses: List[int], pattern_name: str):
    """
    Applies all sampling methods to the given access list and
    displays them in subplots for easy side-by-side comparison.
    Prints coverage stats for each, and finally prints the method
    with the highest coverage.
    """
    sampling_methods = {
        "Uniform": lambda acc: uniform_sampling(acc, int(len(acc)*SAMPLE_PROPORTION)),
        "Random": lambda acc: random_sampling(acc, SAMPLE_PROPORTION),
        "Stratified": lambda acc: stratified_sampling(acc, 10, SAMPLE_PROPORTION),
        "Systematic": lambda acc: systematic_sampling(acc, step=int(1.0/SAMPLE_PROPORTION)),
        "Weighted": lambda acc: weighted_sampling(acc, num_samples=int(len(acc)*SAMPLE_PROPORTION)),
        "Cluster": lambda acc: cluster_sampling(acc, cluster_size=500, num_clusters=2),
        "Temporal": lambda acc: temporal_sampling(acc, window_size=500, sample_proportion=SAMPLE_PROPORTION),
        "Adaptive": lambda acc: adaptive_sampling(acc, threshold=int(MEMORY_SIZE*0.7),
                                                  high_rate=0.5, low_rate=0.1),
        "Reservoir": lambda acc: reservoir_sampling(acc, reservoir_size=int(len(acc)*SAMPLE_PROPORTION)),
        "Frequency": lambda acc: frequency_based_sampling(acc, SAMPLE_PROPORTION),
        "Spatial": lambda acc: spatial_sampling(acc, num_regions=10, sample_proportion=SAMPLE_PROPORTION),
        "Hybrid": lambda acc: hybrid_sampling(acc),
    }

    n_methods = len(sampling_methods)
    fig, axes = plt.subplots(
        nrows=math.ceil(n_methods / 3),
        ncols=3,
        figsize=(15, 4 * math.ceil(n_methods / 3))
    )
    axes = axes.flatten() if n_methods > 1 else [axes]

    time_indices = list(range(len(accesses)))

    coverage_results = []

    for ax_idx, (method_name, sampling_func) in enumerate(sampling_methods.items()):
        if ax_idx >= len(axes):
            break

        # Compute the sample
        sampled_indices, sampled_accesses = sampling_func(accesses)

        # Coverage
        coverage = calculate_coverage(accesses, sampled_accesses)
        coverage_results.append((method_name, coverage))

        # Reuse intervals
        reuse_dict = analyze_temporal_relationships(accesses, sampled_accesses)

        # Plot
        ax = axes[ax_idx]
        ax.plot(time_indices, accesses, '.', alpha=0.2, markersize=3, color='gray', label='All Accesses')
        ax.plot(sampled_indices, sampled_accesses, 'r.', alpha=0.7, markersize=4, label='Sampled')
        ax.set_title(f"{method_name}\nCoverage={coverage:.1f}%, ReuseAddrs={len(reuse_dict)}")
        ax.set_xlabel("Access Index")
        ax.set_ylabel("Address")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for ax_idx in range(n_methods, len(axes)):
        axes[ax_idx].axis('off')

    fig.suptitle(f"Sampling Comparison for Pattern: {pattern_name}", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

    # Determine which method had the highest coverage
    best_method, best_cov = max(coverage_results, key=lambda x: x[1])
    print(f"==> For pattern '{pattern_name}', the highest coverage was {best_cov:.2f}% with method '{best_method}'.\n")

# ---------------------------------------------------------------
#                    MAIN FUNCTION
# ---------------------------------------------------------------
def main():
    # Dictionary of sample patterns
    pattern_functions = {
        "Sequential": sequential_access,
        "Reverse Sequential": reverse_sequential_access,
        "Strided(4)": lambda mem, n: strided_access(mem, n, stride=4),
        "Random": random_access,
        "Bursty": bursty_access,
        "Hotspot": hotspot_access,
        "Periodic(100)": lambda mem, n: periodic_access(mem, n, period=100),
        "Streaming": streaming_access,
    }

    for pattern_name, func in pattern_functions.items():
        print(f"=== Generating Access Pattern: {pattern_name} ===")
        accesses = func(memory_space, NUM_ACCESSES)
        compare_all_sampling_methods_for_pattern(accesses, pattern_name)

if __name__ == "__main__":
    main()
