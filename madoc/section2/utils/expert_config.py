"""
Expert configuration utilities for parallelized Merge_KAN training.

This module provides consistent expert configuration mapping across
Phase 1 (parallel training) and Phase 2 (merging).
"""


def get_expert_configs(n_seeds=5):
    """
    Generate all expert configurations for Merge_KAN.

    Args:
        n_seeds: Number of random seeds per configuration

    Returns:
        List of config dicts with keys: depth, k, seed, grid, epochs
    """
    configs = []

    # Vary depth (2, 3) with k=3 (cubic splines)
    for depth in [2, 3]:
        for seed in range(n_seeds):
            configs.append({
                'depth': depth,
                'k': 3,
                'seed': seed,
                'grid': 5,
                'epochs': 1000  # Default, can be overridden
            })

    # Vary spline order with depth=2
    for k in [2]:  # Quadratic (k=3 covered above)
        for seed in range(n_seeds):
            configs.append({
                'depth': 2,
                'k': k,
                'seed': seed,
                'grid': 5,
                'epochs': 1000  # Default, can be overridden
            })

    return configs


def get_expert_config(index, n_seeds=5, epochs=None):
    """
    Get expert configuration for a specific index (for job arrays).

    Args:
        index: Expert index (0-based)
        n_seeds: Number of random seeds per configuration
        epochs: Override default epochs (default: 1000)

    Returns:
        Config dict with keys: depth, k, seed, grid, epochs

    Raises:
        ValueError: If index is out of range
    """
    configs = get_expert_configs(n_seeds)

    if index < 0 or index >= len(configs):
        raise ValueError(f"Expert index {index} out of range [0, {len(configs)-1}]")

    config = configs[index].copy()

    # Override epochs if specified
    if epochs is not None:
        config['epochs'] = epochs

    return config


def get_num_experts(n_seeds=5):
    """
    Get total number of expert configurations.

    Args:
        n_seeds: Number of random seeds per configuration

    Returns:
        Total number of experts
    """
    return len(get_expert_configs(n_seeds))


def format_expert_name(config, dim):
    """
    Generate consistent expert filename from configuration.

    Args:
        config: Expert config dict
        dim: Problem dimension

    Returns:
        Expert filename (without extension)

    Example:
        format_expert_name({'depth': 2, 'k': 3, 'seed': 0}, 4)
        → "expert_4d_depth2_k3_seed0"
    """
    return f"expert_{dim}d_depth{config['depth']}_k{config['k']}_seed{config['seed']}"


def parse_expert_name(filename):
    """
    Parse expert configuration from filename.

    Args:
        filename: Expert filename (with or without extension)

    Returns:
        Dict with 'dim', 'depth', 'k', 'seed'

    Example:
        parse_expert_name("expert_4d_depth2_k3_seed0.pkl")
        → {'dim': 4, 'depth': 2, 'k': 3, 'seed': 0}
    """
    import re

    # Remove extension if present
    name = filename.split('.')[0]

    # Parse pattern: expert_{dim}d_depth{depth}_k{k}_seed{seed}
    pattern = r'expert_(\d+)d_depth(\d+)_k(\d+)_seed(\d+)'
    match = re.match(pattern, name)

    if not match:
        raise ValueError(f"Invalid expert filename format: {filename}")

    return {
        'dim': int(match.group(1)),
        'depth': int(match.group(2)),
        'k': int(match.group(3)),
        'seed': int(match.group(4))
    }
