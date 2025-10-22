import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from ensemble import KANEnsemble
from utils import save_run

parser = argparse.ArgumentParser(description='Section 2.1: Ensemble of KAN Experts')
parser.add_argument('--epochs', type=int, default=100, help='Epochs (default: 100)')
parser.add_argument('--n_experts', type=int, default=10, help='N experts (default: 10)')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Running {args.n_experts} experts for {args.epochs} epochs")

# Section 2.1: Hierarchical Ensemble of KAN Experts
# ============= Create Dataset =============
print("\n" + "="*60)
print("Section 2.1: KAN Expert Ensemble")
print("="*60)

# Simple synthetic function: f(x0, x1, x2) = 2*x0 + sin(x1) + x2^2
f = lambda x: 2*x[:,[0]] + torch.sin(x[:,[1]]) + x[:,[2]]**2
dataset = create_dataset(f, n_var=3, train_num=500, test_num=200)

# ============= Train Ensemble =============
ensemble = KANEnsemble(
    n_experts=args.n_experts,
    width=16,
    grid=5,
    depth=3,
    device=device
)

results = ensemble.train(dataset, epochs=args.epochs, verbose=True)

# ============= Evaluation =============
print("\n" + "="*60)
print("Ensemble Results")
print("="*60)

X_test = dataset['test_input'].to(device)
y_test = dataset['test_label'].to(device)

# Ensemble predictions with uncertainty
y_pred, uncertainty = ensemble.predict(X_test, uncertainty=True)
mse = torch.nn.functional.mse_loss(y_pred, y_test).item()

print(f"\nIndividual expert losses:")
for i, loss in enumerate(results['losses']):
    print(f"  Expert {i}: {loss:.6f}")
print(f"\nEnsemble (averaging) MSE: {mse:.6f}")
print(f"Mean expert MSE: {results['mean_loss']:.6f}")
print(f"Std expert MSE: {results['std_loss']:.6f}")
print(f"Mean uncertainty: {uncertainty.mean():.6f}")

# Save results
all_results = {
    'expert_losses': results['losses'],
    'ensemble_mse': mse,
    'mean_uncertainty': uncertainty.mean().item(),
    'std_uncertainty': uncertainty.std().item()
}

save_run(all_results, 'section2_1',
         models=results['models'],
         epochs=args.epochs, n_experts=args.n_experts, device=str(device),
         grid=5, width=16, depth=3)

print("\n" + "="*60)
print("Section 2.1 Complete")
print("="*60)
