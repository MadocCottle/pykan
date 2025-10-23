import sys
from pathlib import Path

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from utils import data_funcs as dfs
from utils import save_run, track_time, print_timing_summary
from utils import run_merge_kan_experiment
import argparse
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.3: Merge_KAN Experiments')
parser.add_argument('--n-seeds', type=int, default=5, help='Number of random seeds per expert config (default: 5)')
parser.add_argument('--test-mode', action='store_true', help='Run in test mode with reduced experts (default: False)')
args = parser.parse_args()

n_seeds = args.n_seeds
test_mode = args.test_mode

if test_mode:
    print("Running in TEST MODE with reduced expert pool")
    n_seeds = 2  # Reduce seeds for faster testing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Number of seeds per config: {n_seeds}")

# Section 2.3: Merge_KAN on 2D Poisson PDE
# ============= Create Datasets =============
datasets = []
true_functions = [dfs.f_poisson_2d_sin, dfs.f_poisson_2d_poly, dfs.f_poisson_2d_highfreq, dfs.f_poisson_2d_spec]
dataset_names = ['poisson_2d_sin', 'poisson_2d_poly', 'poisson_2d_highfreq', 'poisson_2d_spec']

for f in true_functions:
    datasets.append(create_dataset(f, n_var=2, train_num=1000, test_num=1000))

print("\n" + "="*80)
print("Starting Section 2.3 - Merge_KAN Experiments")
print("="*80 + "\n")

timers = {}
all_results = []
all_models = {}

# Run Merge_KAN experiment on each dataset
for i, dataset in enumerate(datasets):
    dataset_name = dataset_names[i]
    true_func = true_functions[i]

    def run_experiment():
        return run_merge_kan_experiment(
            dataset, i, dataset_name, device, true_func, n_seeds=n_seeds, verbose=True
        )

    # Track time for this dataset
    result = track_time(timers, f"Merge_KAN {dataset_name}", run_experiment)

    if result is not None:
        all_results.append(result)
        all_models[i] = result['merged_model']

# Print timing summary
print_timing_summary(timers, "Section 2.3", num_datasets=len(datasets))

# ============= Create Summary DataFrames =============

print("\n" + "="*80)
print("MERGE_KAN RESULTS SUMMARY")
print("="*80 + "\n")

# Summary table: one row per dataset
summary_rows = []
for result in all_results:
    summary_rows.append({
        'dataset_idx': result['dataset_idx'],
        'dataset_name': result['dataset_name'],
        'n_experts_trained': len(result['experts']),
        'n_experts_selected': len(result['selected_experts']),
        'merged_dense_mse': result['dense_mse'],
        'merged_num_params': result['num_params'],
        'grids_completed': len(result['grid_history'])
    })

summary_df = pd.DataFrame(summary_rows)
print("\nDataset Summary:")
print(summary_df.to_string(index=False))

# Expert details table: one row per expert
expert_rows = []
for result in all_results:
    for expert in result['experts']:
        expert_rows.append({
            'dataset_idx': result['dataset_idx'],
            'dataset_name': result['dataset_name'],
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
for result in all_results:
    for expert in result['selected_experts']:
        selected_expert_rows.append({
            'dataset_idx': result['dataset_idx'],
            'dataset_name': result['dataset_name'],
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
for result in all_results:
    for grid_info in result['grid_history']:
        grid_rows.append({
            'dataset_idx': result['dataset_idx'],
            'dataset_name': result['dataset_name'],
            'grid_size': grid_info['grid_size'],
            'final_train_loss': grid_info['final_train_loss'],
            'final_test_loss': grid_info['final_test_loss']
        })

grid_df = pd.DataFrame(grid_rows)

# Print comparison with baseline (if available from Section 1.3)
print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)
print("\nMerge_KAN Dense MSE by Dataset:")
for _, row in summary_df.iterrows():
    print(f"  {row['dataset_name']:<25} {row['merged_dense_mse']:.6e} ({row['merged_num_params']} params)")

print("\nNote: Compare these results with Section 1.3 baseline KAN results for evaluation.")

# ============= Save Results =============

# Prepare data for saving
results_dict = {
    'summary': summary_df,
    'experts': expert_df,
    'selected_experts': selected_df,
    'grid_history': grid_df
}

# Prepare metadata
metadata = {
    'n_seeds': n_seeds,
    'test_mode': test_mode,
    'n_datasets': len(datasets),
    'dataset_names': dataset_names
}

save_run(results_dict, 'section2_3',
         models={'merged_kans': all_models},
         epochs=None,  # Not applicable for Merge_KAN
         device=str(device),
         **metadata)

print("\n" + "="*80)
print("Section 2.3 Complete!")
print("="*80)
print(f"\nResults saved to: results/section2_3/")
print(f"  - summary.csv: High-level results per dataset")
print(f"  - experts.csv: All trained experts")
print(f"  - selected_experts.csv: Experts selected for merging")
print(f"  - grid_history.csv: Grid refinement progression")
print(f"  - models.pt: Saved merged KAN models")
