import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from utils import save_run

parser = argparse.ArgumentParser(description='Section 2.5: Pruning and Regularization')
parser.add_argument('--epochs', type=int, default=100, help='Epochs (default: 100)')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Section 2.5: Pruning and Regularization
print("\n" + "="*60)
print("Section 2.5: Pruning and Regularization")
print("="*60)

# Dataset
f = lambda x: 2*x[:,[0]] + torch.sin(x[:,[1]]) + x[:,[2]]**2
dataset = create_dataset(f, n_var=3, train_num=500, test_num=200)

# Train base model
print("\n--- Training base model ---")
model = KAN(width=[3, 16, 16, 1], grid=5, k=3, device=device)
model.fit(dataset, opt='LBFGS', steps=args.epochs, lr=1e-3, lamb=0.01, lamb_entropy=10.0)

# Test base
X_test = dataset['test_input'].to(device)
y_test = dataset['test_label'].to(device)
with torch.no_grad():
    pred_base = model(X_test)
    mse_base = torch.nn.functional.mse_loss(pred_base, y_test).item()

n_params_base = sum(p.numel() for p in model.parameters())

print(f"\nBase model:")
print(f"  Test MSE: {mse_base:.6f}")
print(f"  Parameters: {n_params_base}")

# Prune model
print("\n--- Pruning model ---")
model_pruned = model.prune()
model_pruned.fit(dataset, opt='LBFGS', steps=args.epochs//2, lr=1e-3, lamb=0.01)

# Test pruned
with torch.no_grad():
    pred_pruned = model_pruned(X_test)
    mse_pruned = torch.nn.functional.mse_loss(pred_pruned, y_test).item()

n_params_pruned = sum(p.numel() for p in model_pruned.parameters())

print(f"\nPruned model:")
print(f"  Test MSE: {mse_pruned:.6f}")
print(f"  Parameters: {n_params_pruned}")
print(f"  Reduction: {(1 - n_params_pruned/n_params_base)*100:.1f}%")
print(f"  MSE change: {((mse_pruned - mse_base)/mse_base)*100:.1f}%")

# Save
results = {
    'base': {'mse': mse_base, 'n_params': n_params_base},
    'pruned': {'mse': mse_pruned, 'n_params': n_params_pruned}
}

save_run(results, 'section2_5',
         models={'base': model, 'pruned': model_pruned},
         epochs=args.epochs, device=str(device))

print("\n" + "="*60)
print("Section 2.5 Complete")
print("="*60)
