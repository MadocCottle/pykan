"""
Phase 2: Merge and train combined KAN from pre-trained experts.

This script loads all trained expert models from Phase 1, selects the best
experts per dependency pattern, merges them into a single KAN, and trains
the merged model through grid refinement.

Usage:
    python section2_3_merge.py --dim 4 --expert-dir ./experts_4d --output-dir ./results
    python section2_3_merge.py --dim 10 --expert-dir ./experts_10d --merged-epochs 200
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
from utils.merge_kan import select_best_experts, merge_kans, train_merged_kan_with_refinement
from utils.expert_io import load_all_experts, print_expert_summary
from utils.metrics import count_parameters

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Merge and train KAN from pre-trained experts (Phase 2)')
parser.add_argument('--dim', type=int, required=True, choices=[2, 4, 10],
                    help='Problem dimension (2, 4, or 10)')
parser.add_argument('--expert-dir', type=str, required=True,
                    help='Directory containing trained expert models from Phase 1')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Output directory for saving merged model results (default: section2/results)')
parser.add_argument('--merged-epochs', type=int, default=200,
                    help='Training steps per grid for merged model (default: 200)')
parser.add_argument('--grids', type=int, nargs='+', default=[3, 5, 10, 20],
                    help='Grid sizes for refinement (default: 3 5 10 20)')
args = parser.parse_args()

# Configuration
dim = args.dim
expert_dir = Path(args.expert_dir)
merged_epochs = args.merged_epochs
grids = args.grids

if args.output_dir:
    output_dir = Path(args.output_dir)
else:
    output_dir = Path(__file__).parent / "results"

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Select function and dataset name based on dimension
FUNCTIONS = {
    2: dfs.f_poisson_2d_sin,
    4: dfs.f_poisson_4d_sin,
    10: dfs.f_poisson_10d_sin
}

DATASET_NAMES = {
    2: 'poisson_2d_sin',
    4: 'poisson_4d_sin',
    10: 'poisson_10d_sin'
}

true_function = FUNCTIONS[dim]
dataset_name = DATASET_NAMES[dim]

print("\n" + "="*80)
print(f"Section 2.3 Phase 2: Merge and Train ({dim}D)")
print("="*80)
print(f"Configuration:")
print(f"  Dimension: {dim}D")
print(f"  Dataset: {dataset_name}")
print(f"  Expert dir: {expert_dir}")
print(f"  Output dir: {output_dir}")
print(f"  Merged model epochs/grid: {merged_epochs}")
print(f"  Grid schedule: {grids}")
print()

# Create dataset (same as used for expert training)
print(f"Creating {dim}D dataset...")
dataset = create_dataset(true_function, n_var=dim, train_num=1000, test_num=1000)
print(f"  Train samples: {dataset['train_input'].shape[0]}")
print(f"  Test samples: {dataset['test_input'].shape[0]}")
print()

# =============================================================================
# Load Expert Models
# =============================================================================

print("="*80)
print("Loading Trained Experts")
print("="*80)
print()

timers = {}

experts = track_time(timers, "Loading experts", load_all_experts, expert_dir)

print(f"\nLoaded {len(experts)} expert models")
print()

# Print summary of all experts
for i, expert in enumerate(experts):
    print_expert_summary(expert, name=f"Expert {i}")

# =============================================================================
# Select Best Experts
# =============================================================================

print("\n" + "="*80)
print("Selecting Best Experts per Dependency Pattern")
print("="*80)
print()

selected_experts = select_best_experts(experts)

print(f"\nSelected {len(selected_experts)} experts for merging")
print()

# =============================================================================
# Merge Experts
# =============================================================================

print("="*80)
print("Merging Expert Models")
print("="*80)
print()

expert_models = [e['model'] for e in selected_experts]
expert_dependencies = [e['dependencies'] for e in selected_experts]

def merge_experts():
    return merge_kans(
        expert_models,
        input_dim=dim,
        device=device,
        expert_dependencies=expert_dependencies
    )

merged_model = track_time(timers, "Merging experts", merge_experts)

print(f"\nMerged model architecture: {merged_model.width}")
print(f"Merged model parameters: {count_parameters(merged_model)}")
print()

# =============================================================================
# Train Merged Model
# =============================================================================

print("="*80)
print("Training Merged Model")
print("="*80)
print()

def train_merged():
    return train_merged_kan_with_refinement(
        merged_model,
        dataset,
        device,
        true_function,
        dataset_name=dataset_name,
        grids=grids,
        steps_per_grid=merged_epochs,
        early_stopping=True
    )

training_results = track_time(timers, "Training merged model", train_merged)

# Print timing summary
print_timing_summary(timers, f"Section 2.3 Phase 2 ({dim}D)", num_datasets=1)

# =============================================================================
# Create Summary DataFrames
# =============================================================================

print("\n" + "="*80)
print("MERGE_KAN RESULTS SUMMARY")
print("="*80)
print()

# Summary table
summary_row = {
    'dataset_name': dataset_name,
    'dimension': dim,
    'n_experts_trained': len(experts),
    'n_experts_selected': len(selected_experts),
    'merged_dense_mse': training_results['dense_mse'],
    'merged_num_params': training_results['num_params'],
    'grids_completed': len(training_results['grid_history'])
}

summary_df = pd.DataFrame([summary_row])
print("\nDataset Summary:")
print(summary_df.to_string(index=False))

# Expert details table
expert_rows = []
for expert in experts:
    expert_rows.append({
        'dataset_name': dataset_name,
        'dimension': dim,
        'depth': expert['config']['depth'],
        'k': expert['config']['k'],
        'seed': expert['config']['seed'],
        'dense_mse': expert['dense_mse'],
        'dependencies': str(expert['dependencies']),
        'num_params': expert['num_params'],
        'train_time': expert['train_time'],
        'selected': expert in selected_experts
    })

expert_df = pd.DataFrame(expert_rows)

# Selected experts table
selected_expert_rows = []
for expert in selected_experts:
    selected_expert_rows.append({
        'dataset_name': dataset_name,
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
for grid_info in training_results['grid_history']:
    grid_rows.append({
        'dataset_name': dataset_name,
        'dimension': dim,
        'grid_size': grid_info['grid_size'],
        'final_train_loss': grid_info['final_train_loss'],
        'final_test_loss': grid_info['final_test_loss']
    })

grid_df = pd.DataFrame(grid_rows)

print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)
print(f"\nMerged KAN Dense MSE: {training_results['dense_mse']:.6e}")
print(f"Merged KAN Parameters: {training_results['num_params']}")
print()

# =============================================================================
# Save Results
# =============================================================================

# Prepare data for saving
results_dict = {
    'summary': summary_df,
    'experts': expert_df,
    'selected_experts': selected_df,
    'grid_history': grid_df
}

# Prepare metadata
metadata = {
    'dimension': dim,
    'n_experts_trained': len(experts),
    'n_experts_selected': len(selected_experts),
    'dataset_name': dataset_name,
    'merged_epochs_per_grid': merged_epochs,
    'grid_schedule': grids
}

run_name = f'section2_3_{dim}d'
save_run(results_dict, run_name,
         models={'merged_kan': training_results['model']},
         epochs=None,  # Not applicable for Merge_KAN
         device=str(device),
         **metadata)

print("\n" + "="*80)
print("Phase 2 Complete!")
print("="*80)
print(f"\nResults saved to: results/{run_name}/")
print(f"  - summary.csv: High-level results")
print(f"  - experts.csv: All trained experts")
print(f"  - selected_experts.csv: Experts selected for merging")
print(f"  - grid_history.csv: Grid refinement progression")
print(f"  - merged_kan model: Saved merged KAN")
print()
