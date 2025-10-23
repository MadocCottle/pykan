"""
Merge KAN utilities for Section 2.3.

This module implements functions for training expert KANs, detecting their
functional dependencies, and merging them into a single larger KAN.
"""

import time
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from kan import KAN
from .metrics import count_parameters, dense_mse_error_from_dataset


def detect_dependencies(model, threshold=1e-2):
    """
    Detect which input variables a trained KAN depends on.

    Args:
        model: Trained KAN model
        threshold: Attribution score threshold for considering an input active

    Returns:
        Tuple of active input indices (e.g., (0, 2, 5) means depends on x_0, x_2, x_5)
    """
    # Ensure we have activations cached
    if model.acts is None:
        raise ValueError("Model must have cached activations. Run model.forward() first.")

    # Compute attribution scores
    model.attribute()

    # Get input layer attribution scores
    input_scores = model.node_scores[0]

    # Find active inputs above threshold
    active_mask = input_scores > threshold
    active_inputs = torch.where(active_mask)[0]

    return tuple(active_inputs.tolist())


def train_expert_kan(dataset, config, device, true_function=None, dataset_name="", verbose=True):
    """
    Train a single expert KAN with specified configuration.

    Args:
        dataset: Dataset dict with train/test inputs and labels
        config: Dict with 'depth', 'k', 'seed', 'grid', 'epochs'
        device: Device to train on
        true_function: True function for computing dense MSE
        dataset_name: Name for logging
        verbose: Whether to print progress

    Returns:
        Dict with 'model', 'dense_mse', 'config', 'dependencies', 'train_time'
    """
    n_var = dataset['train_input'].shape[1]
    depth = config.get('depth', 2)
    k = config.get('k', 3)
    seed = config.get('seed', 1)
    grid = config.get('grid', 5)
    epochs = config.get('epochs', 1000)

    # Build width list based on depth
    hidden_size = 5
    width = [n_var] + [hidden_size] * depth + [1]

    if verbose:
        print(f"  Training expert: depth={depth}, k={k}, seed={seed}, width={width}")

    start_time = time.time()

    # Create and train model
    model = KAN(width=width, grid=grid, k=k, seed=seed, device=device)
    # log=1 is minimum (can't be 0), but we can suppress output by setting quiet
    model.fit(dataset, opt="LBFGS", steps=epochs, log=max(1, epochs//10))

    train_time = time.time() - start_time

    # Compute dense MSE
    with torch.no_grad():
        dense_mse = dense_mse_error_from_dataset(model, dataset, true_function,
                                                  num_samples=10000, device=device)

    # Detect dependencies
    model.forward(dataset['train_input'])  # Cache activations
    dependencies = detect_dependencies(model, threshold=1e-2)

    if verbose:
        print(f"    Dense MSE: {dense_mse:.6e}, Dependencies: {dependencies}, Time: {train_time:.1f}s")

    return {
        'model': model,
        'dense_mse': dense_mse,
        'config': config,
        'dependencies': dependencies,
        'train_time': train_time,
        'num_params': count_parameters(model)
    }


def generate_expert_pool(dataset, device, true_function=None, dataset_name="", n_seeds=5):
    """
    Generate a pool of expert KANs with varied configurations.

    Args:
        dataset: Dataset dict
        device: Device to train on
        true_function: True function for metrics
        dataset_name: Name for logging
        n_seeds: Number of random seeds per configuration

    Returns:
        List of expert dicts from train_expert_kan()
    """
    print(f"\nGenerating expert pool for {dataset_name}...")

    # Define configuration variations
    configs = []

    # Vary depth (2, 3, 4) with k=3 (cubic splines)
    for depth in [2, 3]:
        for seed in range(n_seeds):
            configs.append({'depth': depth, 'k': 3, 'seed': seed, 'grid': 5, 'epochs': 1000})

    # Vary spline order with depth=2
    for k in [2]:  # Linear and quadratic (skip k=3 as it's covered above)
        for seed in range(n_seeds):
            configs.append({'depth': 2, 'k': k, 'seed': seed, 'grid': 5, 'epochs': 1000})

    print(f"Total expert configurations: {len(configs)}")

    experts = []
    for i, config in enumerate(configs):
        print(f"\nExpert {i+1}/{len(configs)}:")
        expert = train_expert_kan(dataset, config, device, true_function, dataset_name)
        experts.append(expert)

    return experts


def select_best_experts(experts):
    """
    Select the best expert for each unique dependency pattern.

    Args:
        experts: List of expert dicts from train_expert_kan()

    Returns:
        List of selected expert dicts (best per dependency pattern)
    """
    # Group experts by dependencies
    dependency_groups = {}
    for expert in experts:
        dep_key = expert['dependencies']
        if dep_key not in dependency_groups:
            dependency_groups[dep_key] = []
        dependency_groups[dep_key].append(expert)

    print(f"\nFound {len(dependency_groups)} unique dependency patterns:")
    for dep_key, group in dependency_groups.items():
        print(f"  Dependencies {dep_key}: {len(group)} experts")

    # Select best (lowest dense_mse) per group
    selected_experts = []
    for dep_key, group in dependency_groups.items():
        best = min(group, key=lambda x: x['dense_mse'])
        selected_experts.append(best)
        print(f"  Selected expert with dependencies {dep_key}: dense_mse={best['dense_mse']:.6e}")

    return selected_experts


def merge_kans(expert_models, input_dim, device, grid=None, k=None):
    """
    Merge multiple expert KANs into a single wider KAN.

    Strategy (Option A):
    - Create merged KAN with full input dimension
    - Sum hidden layer widths from all experts
    - Experts' inactive input connections remain masked (set to zero)
    - All expert outputs feed into final aggregation layer

    Args:
        expert_models: List of trained KAN models
        input_dim: Full input dimension
        device: Device for merged model
        grid: Initial grid size for merged model (default: use first expert's grid)
        k: Spline order for merged model (default: use first expert's k)

    Returns:
        Merged KAN model
    """
    if len(expert_models) == 0:
        raise ValueError("Need at least one expert to merge")

    print(f"\nMerging {len(expert_models)} expert KANs...")

    # Use first expert's grid and k if not specified
    if grid is None:
        grid = expert_models[0].grid
    if k is None:
        k = expert_models[0].k

    # Calculate total hidden layer width (sum of all expert hidden widths)
    # We'll merge experts at their first hidden layer
    total_hidden = 0
    for expert in expert_models:
        # Get the first hidden layer size (width[1])
        expert_hidden = expert.width_out[1]
        total_hidden += expert_hidden
        print(f"  Expert with hidden width {expert_hidden}, grid={expert.grid}, k={expert.k}")

    print(f"  Merged hidden layer width: {total_hidden}")
    print(f"  Using grid={grid}, k={k}")

    # Create merged architecture: [input_dim, total_hidden, n_intermediate, 1]
    # Add an intermediate layer to allow learned aggregation
    n_intermediate = max(3, len(expert_models))  # At least 3, or number of experts
    merged = KAN(width=[input_dim, total_hidden, n_intermediate, 1],
                 grid=grid, k=k, seed=0, device=device)

    print(f"  Merged architecture: {merged.width}")

    # Copy weights from experts to merged model's first layer
    # Each expert contributes to a "slice" of the hidden layer
    hidden_offset = 0

    for expert_idx, expert in enumerate(expert_models):
        expert_hidden = expert.width_out[1]
        expert_input = expert.width_in[0]

        # Get the expert's first layer
        expert_layer = expert.act_fun[0]
        merged_layer = merged.act_fun[0]

        # Copy spline coefficients and grid
        # Source: expert_layer has shape (expert_input, expert_hidden, ...)
        # Target: merged_layer slice (input_dim, hidden_offset:hidden_offset+expert_hidden, ...)

        # We need to map expert inputs to full input space
        # For now, assume expert uses first expert_input dimensions
        # TODO: Track actual input mapping from dependency detection

        # Check if expert's grid/k parameters match merged model
        params_match = (expert.grid == merged.grid and expert.k == merged.k)

        if params_match:
            # Direct weight transfer when grid/k match
            for i in range(min(expert_input, input_dim)):
                # Copy grid for this input dimension
                merged_layer.grid.data[i, :] = expert_layer.grid.data[i, :]

                for j in range(expert_hidden):
                    target_j = hidden_offset + j

                    # Copy spline coefficients
                    merged_layer.coef.data[i, target_j, :] = expert_layer.coef.data[i, j, :]

                    # Copy scaling factors
                    merged_layer.scale_base.data[i, target_j] = expert_layer.scale_base.data[i, j]
                    merged_layer.scale_sp.data[i, target_j] = expert_layer.scale_sp.data[i, j]

                    # Set mask to active
                    merged_layer.mask.data[i, target_j] = expert_layer.mask.data[i, j]
        else:
            # If grid/k don't match, just initialize connections and let training adjust
            # Set masks to active for expert's connections
            print(f"    Warning: Expert {expert_idx+1} has different grid/k, will retrain connections")
            for i in range(min(expert_input, input_dim)):
                for j in range(expert_hidden):
                    target_j = hidden_offset + j
                    # Just activate the mask, keep random initialization
                    merged_layer.mask.data[i, target_j] = 1.0

        hidden_offset += expert_hidden
        print(f"  Copied expert {expert_idx+1} to hidden neurons [{hidden_offset-expert_hidden}:{hidden_offset}]")

    print("  Merge complete. Model ready for training.")

    return merged


def train_merged_kan_with_refinement(merged_model, dataset, device, true_function,
                                     dataset_name="", grids=[3, 5, 10, 20],
                                     steps_per_grid=200, early_stopping=True):
    """
    Train merged KAN with grid refinement and optional early stopping.

    Implements hybrid strategy: fixed grid schedule + early stopping on test loss.

    Args:
        merged_model: Merged KAN model
        dataset: Dataset dict
        device: Device
        true_function: True function for metrics
        dataset_name: Name for logging
        grids: List of grid sizes to refine through
        steps_per_grid: Training steps per grid
        early_stopping: Whether to stop if test loss increases

    Returns:
        Dict with training results and metrics
    """
    print(f"\nTraining merged KAN on {dataset_name}...")
    print(f"  Grid schedule: {grids}")
    print(f"  Steps per grid: {steps_per_grid}")

    all_train_losses = []
    all_test_losses = []
    grid_history = []
    best_test_loss = float('inf')
    worse_count = 0

    for i, grid_size in enumerate(grids):
        print(f"\n  Grid {i+1}/{len(grids)}: size={grid_size}")

        # Refine grid (except first iteration)
        if i > 0:
            merged_model = merged_model.refine(grid_size)

        # Train
        results = merged_model.fit(dataset, opt="LBFGS", steps=steps_per_grid, log=1)
        all_train_losses.extend(results['train_loss'])
        all_test_losses.extend(results['test_loss'])

        final_test_loss = results['test_loss'][-1]
        grid_history.append({
            'grid_size': grid_size,
            'final_train_loss': results['train_loss'][-1],
            'final_test_loss': final_test_loss
        })

        # Early stopping check
        if early_stopping and final_test_loss > best_test_loss * 1.05:
            worse_count += 1
            print(f"    Warning: Test loss increased ({final_test_loss:.6e} > {best_test_loss:.6e})")
            if worse_count >= 2:
                print(f"    Early stopping: test loss increased for 2 consecutive grids")
                break
        else:
            best_test_loss = min(best_test_loss, final_test_loss)
            worse_count = 0

    # Final evaluation
    with torch.no_grad():
        final_dense_mse = dense_mse_error_from_dataset(merged_model, dataset, true_function,
                                                        num_samples=10000, device=device)

    print(f"\n  Training complete. Final dense MSE: {final_dense_mse:.6e}")

    return {
        'model': merged_model,
        'train_losses': all_train_losses,
        'test_losses': all_test_losses,
        'dense_mse': final_dense_mse,
        'grid_history': grid_history,
        'num_params': count_parameters(merged_model)
    }


def run_merge_kan_experiment(dataset, dataset_idx, dataset_name, device, true_function,
                             n_seeds=5, verbose=True):
    """
    Run complete Merge_KAN experiment: generate experts, select, merge, train.

    Args:
        dataset: Dataset dict
        dataset_idx: Dataset index for results
        dataset_name: Dataset name for logging
        device: Device to train on
        true_function: True function for metrics
        n_seeds: Number of random seeds per expert configuration
        verbose: Whether to print detailed progress

    Returns:
        Dict with all results and models
    """
    print("\n" + "="*80)
    print(f"MERGE_KAN EXPERIMENT: {dataset_name}")
    print("="*80)

    # Phase 1: Generate expert pool
    experts = generate_expert_pool(dataset, device, true_function, dataset_name, n_seeds)

    # Phase 2: Select best per dependency pattern
    selected_experts = select_best_experts(experts)

    if len(selected_experts) == 0:
        print("ERROR: No experts selected!")
        return None

    # Phase 3: Merge experts
    n_var = dataset['train_input'].shape[1]
    expert_models = [e['model'] for e in selected_experts]
    # Use grid/k from first expert (will be consistent if all experts trained with same config)
    merged_model = merge_kans(expert_models, input_dim=n_var, device=device)

    # Phase 4: Train merged model
    training_results = train_merged_kan_with_refinement(
        merged_model, dataset, device, true_function, dataset_name,
        grids=[3, 5, 10, 20], steps_per_grid=200, early_stopping=True
    )

    return {
        'dataset_idx': dataset_idx,
        'dataset_name': dataset_name,
        'experts': experts,
        'selected_experts': selected_experts,
        'merged_model': training_results['model'],
        'train_losses': training_results['train_losses'],
        'test_losses': training_results['test_losses'],
        'dense_mse': training_results['dense_mse'],
        'grid_history': training_results['grid_history'],
        'num_params': training_results['num_params']
    }
