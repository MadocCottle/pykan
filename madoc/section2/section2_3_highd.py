"""
Section 2.3 High-D: Merge_KAN on Higher-Dimensional Poisson PDEs

This script runs Merge_KAN experiments on higher-dimensional problems (4D, 10D).
Unlike the parallelized version, this runs sequentially for simpler execution.

Usage:
    python section2_3_highd.py --dim 4 --n-seeds 5
    python section2_3_highd.py --dim 10 --n-seeds 3 --expert-epochs 500
"""

import sys
from pathlib import Path
import argparse
import pandas as pd

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from utils import data_funcs as dfs
from utils import save_run, track_time, print_timing_summary
from utils import run_merge_kan_experiment

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.3 High-D: Merge_KAN on Higher-Dimensional Poisson PDEs')
parser.add_argument('--dim', type=int, required=True, choices=[4, 10],
                    help='Dimension of the Poisson PDE (4 or 10)')
parser.add_argument('--n-seeds', type=int, default=5,
                    help='Number of random seeds per expert config (default: 5)')
parser.add_argument('--expert-epochs', type=int, default=1000,
                    help='Training epochs per expert (default: 1000)')
parser.add_argument('--merged-epochs', type=int, default=200,
                    help='Training steps per grid for merged model (default: 200)')
parser.add_argument('--test-mode', action='store_true',
                    help='Run in test mode with reduced seeds (default: False)')
args = parser.parse_args()

dim = args.dim
n_seeds = args.n_seeds
expert_epochs = args.expert_epochs
merged_epochs = args.merged_epochs
test_mode = args.test_mode

if test_mode:
    print("Running in TEST MODE with reduced expert pool")
    n_seeds = 2  # Reduce seeds for faster testing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Running {dim}D Merge_KAN with {n_seeds} seeds per config")
print(f"Expert epochs: {expert_epochs}, Merged epochs/grid: {merged_epochs}")

# Select function and dataset name based on dimension
FUNCTIONS = {
    4: dfs.f_poisson_4d_sin,
    10: dfs.f_poisson_10d_sin
}

DATASET_NAMES = {
    4: 'poisson_4d_sin',
    10: 'poisson_10d_sin'
}

# Grid sizes (reduced for higher dimensions to manage computational cost)
GRIDS = {
    4: [3, 5, 10, 20],
    10: [3, 5, 10, 20]
}

true_function = FUNCTIONS[dim]
dataset_name = DATASET_NAMES[dim]
grids = GRIDS[dim]

print(f"\nConfiguration:")
print(f"  Dimension: {dim}D")
print(f"  Function: {dataset_name}")
print(f"  Grid schedule: {grids}")
print(f"  N_SEEDS: {n_seeds}")

# Create Dataset
print(f"\nCreating {dim}D dataset...")
dataset = create_dataset(true_function, n_var=dim, train_num=1000, test_num=1000)

print("\n" + "="*80)
print(f"Starting Section 2.3 High-D - Merge_KAN Experiments ({dim}D)")
print("="*80 + "\n")

timers = {}

# Run Merge_KAN experiment
def run_experiment():
    # Note: run_merge_kan_experiment needs to be updated to accept epochs parameters
    # For now, it will use hardcoded values from merge_kan.py
    # TODO: Update run_merge_kan_experiment to accept expert_epochs and merged_epochs
    return run_merge_kan_experiment(
        dataset, 0, dataset_name, device, true_function,
        n_seeds=n_seeds, verbose=True
    )

# Track time
result = track_time(timers, f"Merge_KAN {dataset_name}", run_experiment)

# Print timing summary
print_timing_summary(timers, f"Section 2.3 High-D ({dim}D)", num_datasets=1)

# Create Summary DataFrames
print("\n" + "="*80)
print("MERGE_KAN RESULTS SUMMARY")
print("="*80 + "\n")

if result is not None:
    # Summary table
    summary_row = {
        'dataset_name': result['dataset_name'],
        'dimension': dim,
        'n_experts_trained': len(result['experts']),
        'n_experts_selected': len(result['selected_experts']),
        'merged_dense_mse': result['dense_mse'],
        'merged_num_params': result['num_params'],
        'grids_completed': len(result['grid_history'])
    }

    summary_df = pd.DataFrame([summary_row])
    print("\nDataset Summary:")
    print(summary_df.to_string(index=False))

    # Expert details table
    expert_rows = []
    for expert in result['experts']:
        expert_rows.append({
            'dataset_name': result['dataset_name'],
            'dimension': dim,
            'depth': expert['config']['depth'],
            'k': expert['config']['k'],
            'seed': expert['config']['seed'],
            'dense_mse': expert['dense_mse'],
            'dependencies': str(expert['dependencies']),
            'num_params': expert['num_params'],
            'train_time': expert['train_time'],
            'selected': expert in result['selected_experts']
        })

    expert_df = pd.DataFrame(expert_rows)

    # Selected experts table
    selected_expert_rows = []
    for expert in result['selected_experts']:
        selected_expert_rows.append({
            'dataset_name': result['dataset_name'],
            'dimension': dim,
            'dependencies': str(expert['dependencies']),
            'dense_mse': expert['dense_mse'],
            'depth': expert['config']['depth'],
            'k': expert['config']['k'],
            'seed': expert['config']['seed']
        })

    selected_df = pd.DataFrame(selected_expert_rows)
    print("\nSelected Experts (Best per Dependency Pattern):")
    print(selected_df.to_string(index=False))

    # Grid refinement history
    grid_rows = []
    for grid_info in result['grid_history']:
        grid_rows.append({
            'dataset_name': result['dataset_name'],
            'dimension': dim,
            'grid_size': grid_info['grid_size'],
            'final_train_loss': grid_info['final_train_loss'],
            'final_test_loss': grid_info['final_test_loss']
        })

    grid_df = pd.DataFrame(grid_rows)

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\nMerged KAN Dense MSE: {result['dense_mse']:.6e}")
    print(f"Merged KAN Parameters: {result['num_params']}")

    # Save Results
    results_dict = {
        'summary': summary_df,
        'experts': expert_df,
        'selected_experts': selected_df,
        'grid_history': grid_df
    }

    metadata = {
        'dimension': dim,
        'n_seeds': n_seeds,
        'test_mode': test_mode,
        'expert_epochs': expert_epochs,
        'merged_epochs': merged_epochs,
        'dataset_name': dataset_name
    }

    run_name = f'section2_3_highd_{dim}d'
    save_run(results_dict, run_name,
             models={'merged_kan': result['merged_model']},
             epochs=None,  # Not applicable for Merge_KAN
             device=str(device),
             **metadata)

    print("\n" + "="*80)
    print("Section 2.3 High-D Complete!")
    print("="*80)
    print(f"\nResults saved to: results/{run_name}/")
    print(f"  - summary.csv: High-level results")
    print(f"  - experts.csv: All trained experts")
    print(f"  - selected_experts.csv: Experts selected for merging")
    print(f"  - grid_history.csv: Grid refinement progression")
    print(f"  - merged_kan model: Saved merged KAN")
else:
    print("ERROR: Experiment failed to produce results")
    sys.exit(1)
