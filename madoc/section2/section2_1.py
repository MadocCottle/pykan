import sys
from pathlib import Path

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from utils import data_funcs as dfs
from utils import save_run, track_time, print_timing_summary, print_optimizer_summary
from utils import run_kan_optimizer_tests, run_kan_lm_tests
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.1: Optimizer Comparison')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
args = parser.parse_args()

epochs = args.epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(device)
print(f"Running with {epochs} epochs")

# Section 2.1: Optimizer Comparison on 2D Poisson PDE
# ============= Create Datasets =============
datasets = []
true_functions = [dfs.f_poisson_2d_sin, dfs.f_poisson_2d_poly, dfs.f_poisson_2d_highfreq, dfs.f_poisson_2d_spec]
dataset_names = ['poisson_2d_sin', 'poisson_2d_poly', 'poisson_2d_highfreq', 'poisson_2d_spec']

for f in true_functions:
    datasets.append(create_dataset(f, n_var=2, train_num=1000, test_num=1000))

grids = np.array([3, 5, 10, 20, 50, 100])

print("\n" + "="*60)
print("Starting Section 2.1 Optimizer Comparison")
print("="*60 + "\n")

timers = {}

# Training with different optimizers
print("Training KANs with Adam optimizer (with dense MSE metrics)...")
adam_results, adam_models = track_time(timers, "KAN Adam training",
                                       run_kan_optimizer_tests,
                                       datasets, grids, epochs, device, "Adam", true_functions, dataset_names)

print("\nTraining KANs with LBFGS optimizer (with dense MSE metrics)...")
lbfgs_results, lbfgs_models = track_time(timers, "KAN LBFGS training",
                                        run_kan_optimizer_tests,
                                        datasets, grids, epochs, device, "LBFGS", true_functions, dataset_names)

print("\nTraining KANs with LM optimizer (with dense MSE metrics)...")
lm_results, lm_models = track_time(timers, "KAN LM training",
                                    run_kan_lm_tests,
                                    datasets, grids, epochs, device, true_functions, dataset_names)

# Print timing summary
print_timing_summary(timers, "Section 2.1", num_datasets=len(datasets))

all_results = {'adam': adam_results, 'lbfgs': lbfgs_results, 'lm': lm_results}
print(f"\nResults summary:")
for model_type, df in all_results.items():
    print(f"  {model_type}: {df.shape[0]} rows, {df.shape[1]} columns")

# Print optimizer summary table
print_optimizer_summary(all_results, dataset_names)

save_run(all_results, 'section2_1',
         models={'adam': adam_models, 'lbfgs': lbfgs_models, 'lm': lm_models},
         epochs=epochs, device=str(device))
# Note: Derivable metadata (grids, num_datasets, dataset_names) can be obtained from DataFrames:
# - grids: adam_results['grid_size'].unique()
# - num_datasets: adam_results['dataset_idx'].nunique()
# - dataset_names: adam_results['dataset_name'].unique()
