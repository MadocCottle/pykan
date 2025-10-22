import sys
from pathlib import Path

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from utils import data_funcs as dfs
from utils import save_run, track_time, print_timing_summary, print_optimizer_summary
from utils import run_kan_adaptive_density_test, run_kan_baseline_test
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.2: Adaptive Density')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
args = parser.parse_args()

epochs = args.epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(device)
print(f"Running with {epochs} epochs")

# Section 2.2: Adaptive Density on 2D Poisson PDE
# ============= Create Datasets =============
datasets = []
true_functions = [dfs.f_poisson_2d_sin, dfs.f_poisson_2d_poly, dfs.f_poisson_2d_highfreq, dfs.f_poisson_2d_spec]
dataset_names = ['poisson_2d_sin', 'poisson_2d_poly', 'poisson_2d_highfreq', 'poisson_2d_spec']

for f in true_functions:
    datasets.append(create_dataset(f, n_var=2, train_num=1000, test_num=1000))

grids = np.array([3, 5, 10, 20, 50, 100])

print("\n" + "="*60)
print("Starting Section 2.2 Adaptive Density")
print("="*60 + "\n")

timers = {}

print("Test 1: Training KANs with adaptive density (alternative to regular densification)...")
adaptive_only_results, adaptive_only_models = track_time(
    timers, "KAN adaptive density only",
    run_kan_adaptive_density_test,
    datasets, grids, epochs, device, False, 1e-2, true_functions, dataset_names
)

print("\nTest 2: Training KANs with adaptive + regular densification...")
adaptive_regular_results, adaptive_regular_models = track_time(
    timers, "KAN adaptive + regular density",
    run_kan_adaptive_density_test,
    datasets, grids, epochs, device, True, 1e-2, true_functions, dataset_names
)

print("\nTraining baseline KANs (regular refinement only for comparison)...")
baseline_results, baseline_models = track_time(
    timers, "KAN baseline",
    run_kan_baseline_test,
    datasets, grids, epochs, device, true_functions, dataset_names
)

# Print timing summary
print_timing_summary(timers, "Section 2.2", num_datasets=len(datasets))

all_results = {
    'adaptive_only': adaptive_only_results,
    'adaptive_regular': adaptive_regular_results,
    'baseline': baseline_results
}

print(f"\nResults summary:")
for model_type, df in all_results.items():
    print(f"  {model_type}: {df.shape[0]} rows, {df.shape[1]} columns")

# Print approach summary table
print_optimizer_summary(all_results, dataset_names)

save_run(all_results, 'section2_2',
         models={
             'adaptive_only': adaptive_only_models,
             'adaptive_regular': adaptive_regular_models,
             'baseline': baseline_models
         },
         epochs=epochs, device=str(device))
