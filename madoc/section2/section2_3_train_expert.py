"""
Phase 1: Train a single expert KAN for parallelized Merge_KAN.

This script is designed to be run in a PBS job array where each job
trains one expert with a specific configuration.

Usage:
    python section2_3_train_expert.py --index 0 --dim 4 --n-seeds 5 --output-dir ./experts
    python section2_3_train_expert.py --index 5 --dim 10 --epochs 500 --output-dir ./experts
"""

import sys
from pathlib import Path
import argparse

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from utils import data_funcs as dfs
from utils.merge_kan import train_expert_kan
from utils.expert_config import get_expert_config, format_expert_name
from utils.expert_io import save_expert, print_expert_summary

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a single expert KAN for Merge_KAN (Phase 1)')
parser.add_argument('--index', type=int, required=True,
                    help='Expert index (0-based, used in job arrays)')
parser.add_argument('--dim', type=int, required=True, choices=[2, 4, 10],
                    help='Problem dimension (2, 4, or 10)')
parser.add_argument('--n-seeds', type=int, default=5,
                    help='Number of random seeds (determines total number of experts, default: 5)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of training epochs (default: 1000)')
parser.add_argument('--output-dir', type=str, required=True,
                    help='Output directory for saving trained expert model')
args = parser.parse_args()

# Configuration
index = args.index
dim = args.dim
n_seeds = args.n_seeds
epochs = args.epochs
output_dir = Path(args.output_dir)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Get expert configuration for this index
try:
    config = get_expert_config(index, n_seeds=n_seeds, epochs=epochs)
except ValueError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print("\n" + "="*80)
print(f"Training Expert {index} for {dim}D Merge_KAN")
print("="*80)
print(f"Configuration:")
print(f"  Index: {index}")
print(f"  Dimension: {dim}D")
print(f"  Depth: {config['depth']}")
print(f"  Spline order (k): {config['k']}")
print(f"  Random seed: {config['seed']}")
print(f"  Grid size: {config['grid']}")
print(f"  Epochs: {config['epochs']}")
print(f"  Output dir: {output_dir}")
print()

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

# Create dataset
print(f"Creating {dim}D dataset ({dataset_name})...")
dataset = create_dataset(true_function, n_var=dim, train_num=1000, test_num=1000)
print(f"  Train samples: {dataset['train_input'].shape[0]}")
print(f"  Test samples: {dataset['test_input'].shape[0]}")
print()

# Train expert
print(f"Training expert {index}...")
print(f"  Config: depth={config['depth']}, k={config['k']}, seed={config['seed']}")
print()

expert_dict = train_expert_kan(
    dataset=dataset,
    config=config,
    device=device,
    true_function=true_function,
    dataset_name=dataset_name,
    verbose=True
)

# Print summary
print()
print("="*80)
print("Expert Training Complete")
print("="*80)
print_expert_summary(expert_dict, name=f"Expert {index}")

# Save expert
expert_filename = format_expert_name(config, dim)
saved_path = save_expert(expert_dict, output_dir, expert_filename)

print()
print(f"Expert saved to: {saved_path}")
print()
print("="*80)
print(f"Expert {index} Complete!")
print("="*80)
