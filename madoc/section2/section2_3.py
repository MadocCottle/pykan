import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from utils import save_run

parser = argparse.ArgumentParser(description='Section 2.3: Multi-Grid Comparison')
parser.add_argument('--epochs', type=int, default=100, help='Epochs (default: 100)')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Section 2.3: Multi-Grid Comparison
print("\n" + "="*60)
print("Section 2.3: Multi-Grid Comparison")
print("="*60)

# Dataset
f = lambda x: 2*x[:,[0]] + torch.sin(x[:,[1]]) + x[:,[2]]**2
dataset = create_dataset(f, n_var=3, train_num=500, test_num=200)

# Compare different grid sizes
grids_to_test = [3, 5, 10, 20]
results = {}

for grid in grids_to_test:
    print(f"\n--- Testing grid={grid} ---")

    model = KAN(width=[3, 16, 16, 1], grid=grid, k=3, device=device)
    model.fit(dataset, opt='LBFGS', steps=args.epochs, lr=1e-3, lamb=0.01)

    # Test
    X_test = dataset['test_input'].to(device)
    y_test = dataset['test_label'].to(device)
    with torch.no_grad():
        pred = model(X_test)
        mse = torch.nn.functional.mse_loss(pred, y_test).item()

    results[f'grid_{grid}'] = {'mse': mse, 'n_params': sum(p.numel() for p in model.parameters())}
    print(f"Test MSE: {mse:.6f}, Params: {results[f'grid_{grid}']['n_params']}")

# Save
save_run(results, 'section2_3',
         epochs=args.epochs, device=str(device),
         grids=grids_to_test)

print("\n" + "="*60)
print("Section 2.3 Complete")
print("="*60)
