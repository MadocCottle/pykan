import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from utils import save_run

parser = argparse.ArgumentParser(description='Section 2.2: Adaptive Grid Densification')
parser.add_argument('--epochs', type=int, default=100, help='Epochs (default: 100)')
parser.add_argument('--initial_grid', type=int, default=3, help='Initial grid (default: 3)')
parser.add_argument('--max_grid', type=int, default=10, help='Max grid (default: 10)')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Section 2.2: Adaptive Grid Densification
print("\n" + "="*60)
print("Section 2.2: Adaptive Grid Densification")
print("="*60)

# Dataset
f = lambda x: 2*x[:,[0]] + torch.sin(x[:,[1]]) + x[:,[2]]**2
dataset = create_dataset(f, n_var=3, train_num=500, test_num=200)

# Train with progressive grid refinement
print(f"\nTraining with grid progression: {args.initial_grid} -> {args.max_grid}")

grids = [args.initial_grid, 5, args.max_grid]
model = KAN(width=[3, 16, 16, 1], grid=grids[0], k=3, device=device)

results = {}
for i, grid in enumerate(grids):
    print(f"\n--- Grid {grid} ---")
    model.fit(dataset, opt='LBFGS', steps=args.epochs//len(grids), lr=1e-3, lamb=0.01)

    # Test
    X_test = dataset['test_input'].to(device)
    y_test = dataset['test_label'].to(device)
    with torch.no_grad():
        pred = model(X_test)
        mse = torch.nn.functional.mse_loss(pred, y_test).item()

    results[f'grid_{grid}'] = {'mse': mse}
    print(f"Test MSE: {mse:.6f}")

    # Refine grid if not last
    if i < len(grids) - 1:
        model = model.refine(grids[i+1])

# Save
save_run(results, 'section2_2',
         models=model,
         epochs=args.epochs, device=str(device),
         initial_grid=args.initial_grid, max_grid=args.max_grid)

print("\n" + "="*60)
print("Section 2.2 Complete")
print("="*60)
