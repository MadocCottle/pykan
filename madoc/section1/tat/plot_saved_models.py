"""Load and plot saved KAN models from section1_1.py"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from kan import *
import data_funcs as dfs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results_dir = Path(__file__).parent / 'sec1_results'

model_dirs = sorted(results_dir.glob('kan_models_*'))
if not model_dirs:
    raise FileNotFoundError("No saved KAN models found. Run section1_1.py first.")

latest_models_dir = model_dirs[-1]
with open(latest_models_dir / 'models_metadata.json', 'r') as f:
    models_metadata = json.load(f)

print(f"Found {models_metadata['num_models']} models in {latest_models_dir}")

freq = [1, 2, 3, 4, 5]
datasets = [create_dataset(dfs.sinusoid_1d(f), n_var=1, train_num=1000, test_num=1000) for f in freq]
datasets.extend([create_dataset(dfs.f_piecewise, n_var=1, train_num=1000, test_num=1000),
                 create_dataset(dfs.f_sawtooth, n_var=1, train_num=1000, test_num=1000),
                 create_dataset(dfs.f_polynomial, n_var=1, train_num=1000, test_num=1000),
                 create_dataset(dfs.f_poisson_1d_highfreq, n_var=1, train_num=1000, test_num=1000)])

dataset_names = ["sin(1x)", "sin(2x)", "sin(3x)", "sin(4x)", "sin(5x)",
                 "piecewise", "sawtooth", "polynomial", "poisson_1d"]

for dataset_idx in models_metadata['dataset_indices']:
    print(f"Plotting dataset {dataset_idx}: {dataset_names[dataset_idx]}")
    model = KAN.loadckpt(str(latest_models_dir / f'kan_dataset_{dataset_idx}'))
    model(datasets[dataset_idx]['train_input'])
    plot_dir = latest_models_dir / f'plots_dataset_{dataset_idx}'
    plot_dir.mkdir(exist_ok=True)
    model.plot(folder=str(plot_dir), beta=100, scale=0.5)

print(f"Plots saved to {latest_models_dir}")

pruned_model_dirs = sorted(results_dir.glob('kan_pruned_models_*'))
if pruned_model_dirs:
    latest_pruned_dir = pruned_model_dirs[-1]
    with open(latest_pruned_dir / 'pruned_models_metadata.json', 'r') as f:
        pruned_metadata = json.load(f)

    print(f"\nFound {pruned_metadata['num_models']} pruned models (params: {pruned_metadata['pruning_params']})")
    for dataset_idx in pruned_metadata['dataset_indices']:
        print(f"Plotting pruned dataset {dataset_idx}: {dataset_names[dataset_idx]}")
        pruned_model = KAN.loadckpt(str(latest_pruned_dir / f'kan_pruned_dataset_{dataset_idx}'))
        pruned_model(datasets[dataset_idx]['train_input'])
        plot_dir = latest_pruned_dir / f'plots_pruned_dataset_{dataset_idx}'
        plot_dir.mkdir(exist_ok=True)
        pruned_model.plot(folder=str(plot_dir), beta=100, scale=0.5)
    print(f"Pruned plots saved to {latest_pruned_dir}")
