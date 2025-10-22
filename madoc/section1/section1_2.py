import sys
from pathlib import Path

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from utils import data_funcs as dfs
from utils import run_mlp_tests, run_siren_tests, run_kan_grid_tests, save_run
from utils import track_time, print_timing_summary, print_best_dense_mse_summary
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 1.2: 1D Poisson PDE')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
args = parser.parse_args()

epochs = args.epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(device)
print(f"Running with {epochs} epochs")

#Section 1.2: 1D Poisson PDE
# ============= Create Datasets =============
datasets = []
true_functions = []
dataset_names = ['poisson_1d_sin', 'poisson_1d_poly', 'poisson_1d_highfreq']

for i, f in enumerate(dfs.sec1_2):
    datasets.append(create_dataset(f, n_var=1, train_num=1000, test_num=1000))
    true_functions.append(f)

grids = np.array([3,5,10,20,50,100])
depths = [2, 3, 4, 5, 6]
activations = ['tanh', 'relu', 'silu']

print("\n" + "="*60)
print("Starting Section 1.2 Training")
print("="*60 + "\n")

# Track timing for all model types
timers = {}

print("\nTraining MLPs (with dense MSE metrics)...")
mlp_results, mlp_models = track_time(timers, "MLP training", run_mlp_tests, datasets, depths, activations, epochs, device, true_functions, dataset_names)

print("Training SIRENs (with dense MSE metrics)...")
siren_results, siren_models = track_time(timers, "SIREN training", run_siren_tests, datasets, depths, epochs, device, true_functions, dataset_names)

print("Training KANs (with dense MSE metrics)...")
kan_results, kan_models = track_time(timers, "KAN training", run_kan_grid_tests, datasets, grids, epochs, device, False, true_functions, dataset_names)

print("Training KANs with pruning (with dense MSE metrics)...")
kan_pruning_results, _, kan_pruned_models = track_time(timers, "KAN pruning training", run_kan_grid_tests, datasets, grids, epochs, device, True, true_functions, dataset_names)

# Print timing summary
print_timing_summary(timers, "Section 1.2", num_datasets=len(datasets))

all_results = {'mlp': mlp_results, 'siren': siren_results, 'kan': kan_results, 'kan_pruning': kan_pruning_results}
print(f"\nResults summary:")
for model_type, df in all_results.items():
    print(f"  {model_type}: {df.shape[0]} rows, {df.shape[1]} columns")

# Print best dense MSE summary table
print_best_dense_mse_summary(all_results, dataset_names)

save_run(all_results, 'section1_2',
         models={'mlp': mlp_models, 'siren': siren_models,
                 'kan': kan_models, 'kan_pruned': kan_pruned_models},
         epochs=epochs, device=str(device))
# Note: Derivable metadata (grids, depths, activations) can be obtained from DataFrames:
# - grids: kan_results['grid_size'].unique()
# - depths: mlp_results['depth'].unique()
# - activations: mlp_results['activation'].unique()
# - num_datasets: mlp_results['dataset_idx'].nunique()