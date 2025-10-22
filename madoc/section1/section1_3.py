import sys
from pathlib import Path

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from utils import data_funcs as dfs
from utils import run_mlp_tests, run_siren_tests, run_kan_grid_tests, save_run
from utils import track_time, print_timing_summary
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 1.3: 2D Poisson PDE')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
args = parser.parse_args()

epochs = args.epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(device)
print(f"Running with {epochs} epochs")

#Section 1.3: 2D Poisson PDE
# ============= Create Datasets =============
datasets = []
true_functions = [dfs.f_poisson_2d_sin, dfs.f_poisson_2d_poly, dfs.f_poisson_2d_highfreq, dfs.f_poisson_2d_spec]

for f in true_functions:
    datasets.append(create_dataset(f, n_var=2, train_num=1000, test_num=1000))

grids = np.array([3,5,10,20,50,100])
depths = [2, 3, 4, 5, 6]
activations = ['tanh', 'relu', 'silu']

print("\n" + "="*60)
print("Starting Section 1.3 Training")
print("="*60 + "\n")

# Track timing for all model types
timers = {}

print("\nTraining MLPs (with dense MSE metrics)...")
mlp_results = track_time(timers, "MLP training", run_mlp_tests, datasets, depths, activations, epochs, device, true_functions, True)

print("Training SIRENs (with dense MSE metrics)...")
siren_results = track_time(timers, "SIREN training", run_siren_tests, datasets, depths, epochs, device, true_functions, True)

print("Training KANs (with dense MSE metrics)...")
kan_results, kan_models = track_time(timers, "KAN training", run_kan_grid_tests, datasets, grids, epochs, device, False, true_functions, True)

# Print timing summary
print_timing_summary(timers, "Section 1.3", num_datasets=len(datasets))

all_results = {'mlp': mlp_results, 'siren': siren_results, 'kan': kan_results}
print(all_results)

save_run(all_results, 'section1_3',
         models={'kan': kan_models},
         epochs=epochs, device=str(device), grids=grids.tolist(),
         depths=depths, activations=activations, num_datasets=len(datasets))