"""
Expert model I/O utilities for parallelized Merge_KAN training.

This module handles saving and loading of trained expert KAN models
with their metadata for Phase 1 and Phase 2 of parallel training.
"""

import pickle
from pathlib import Path
import torch


def save_expert(expert_dict, output_dir, filename):
    """
    Save trained expert model and metadata to disk.

    Args:
        expert_dict: Expert dict from train_expert_kan() with keys:
                    'model', 'dense_mse', 'config', 'dependencies', 'train_time', 'num_params'
        output_dir: Directory to save expert model
        filename: Filename (without extension, .pkl will be added)

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{filename}.pkl"

    # Save the entire expert dict (includes model and metadata)
    with open(filepath, 'wb') as f:
        pickle.dump(expert_dict, f)

    return filepath


def load_expert(filepath):
    """
    Load trained expert model and metadata from disk.

    Args:
        filepath: Path to expert .pkl file

    Returns:
        Expert dict with keys: 'model', 'dense_mse', 'config', 'dependencies', 'train_time', 'num_params'
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Expert file not found: {filepath}")

    with open(filepath, 'rb') as f:
        expert_dict = pickle.load(f)

    return expert_dict


def load_all_experts(expert_dir, pattern="expert_*.pkl"):
    """
    Load all expert models from a directory.

    Args:
        expert_dir: Directory containing expert .pkl files
        pattern: Glob pattern for expert files (default: "expert_*.pkl")

    Returns:
        List of expert dicts, sorted by filename

    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no experts found
    """
    expert_dir = Path(expert_dir)

    if not expert_dir.exists():
        raise FileNotFoundError(f"Expert directory not found: {expert_dir}")

    expert_files = sorted(expert_dir.glob(pattern))

    if len(expert_files) == 0:
        raise ValueError(f"No expert files found in {expert_dir} matching pattern '{pattern}'")

    experts = []
    for filepath in expert_files:
        expert = load_expert(filepath)
        experts.append(expert)

    print(f"Loaded {len(experts)} expert models from {expert_dir}")

    return experts


def get_expert_summary(expert_dict):
    """
    Get a human-readable summary of an expert model.

    Args:
        expert_dict: Expert dict from train_expert_kan()

    Returns:
        Dict with summary information
    """
    config = expert_dict['config']
    return {
        'depth': config['depth'],
        'k': config['k'],
        'seed': config['seed'],
        'grid': config['grid'],
        'dense_mse': expert_dict['dense_mse'],
        'dependencies': expert_dict['dependencies'],
        'num_params': expert_dict['num_params'],
        'train_time': expert_dict.get('train_time', None)
    }


def print_expert_summary(expert_dict, name="Expert"):
    """
    Print a human-readable summary of an expert model.

    Args:
        expert_dict: Expert dict from train_expert_kan()
        name: Name for the expert (default: "Expert")
    """
    summary = get_expert_summary(expert_dict)
    print(f"\n{name}:")
    print(f"  Config: depth={summary['depth']}, k={summary['k']}, seed={summary['seed']}, grid={summary['grid']}")
    print(f"  Dense MSE: {summary['dense_mse']:.6e}")
    print(f"  Dependencies: {summary['dependencies']}")
    print(f"  Parameters: {summary['num_params']}")
    if summary['train_time'] is not None:
        print(f"  Train Time: {summary['train_time']:.1f}s")
