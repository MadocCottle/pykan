import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
from ensemble import KANEnsemble
from utils import save_run

parser = argparse.ArgumentParser(description='Section 2.4: Ensemble Uncertainty')
parser.add_argument('--epochs', type=int, default=100, help='Epochs (default: 100)')
parser.add_argument('--n_experts', type=int, default=5, help='N experts (default: 5)')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Section 2.4: Ensemble Uncertainty Quantification
print("\n" + "="*60)
print("Section 2.4: Ensemble Uncertainty Quantification")
print("="*60)

# Dataset with noise
f = lambda x: 2*x[:,[0]] + torch.sin(x[:,[1]]) + x[:,[2]]**2
dataset = create_dataset(f, n_var=3, train_num=500, test_num=200, noise_scale=0.1)

# Train ensemble
ensemble = KANEnsemble(
    n_experts=args.n_experts,
    width=16,
    grid=5,
    depth=3,
    device=device
)

results = ensemble.train(dataset, epochs=args.epochs, verbose=True)

# Analyze uncertainty
X_test = dataset['test_input'].to(device)
y_test = dataset['test_label'].to(device)

y_pred, uncertainty = ensemble.predict(X_test, uncertainty=True)
mse = torch.nn.functional.mse_loss(y_pred, y_test).item()

# Compute calibration metrics
abs_errors = (y_pred - y_test).abs()
correlation = torch.corrcoef(torch.stack([uncertainty.squeeze(), abs_errors.squeeze()]))[0, 1].item()

print("\n" + "="*60)
print("Uncertainty Analysis")
print("="*60)
print(f"\nEnsemble MSE: {mse:.6f}")
print(f"Mean uncertainty: {uncertainty.mean():.6f}")
print(f"Uncertainty-error correlation: {correlation:.3f}")
print(f"High uncertainty samples (top 10%):")
high_unc_idx = uncertainty.squeeze().argsort(descending=True)[:len(X_test)//10]
print(f"  Mean error: {abs_errors[high_unc_idx].mean():.6f}")
print(f"Low uncertainty samples (bottom 10%):")
low_unc_idx = uncertainty.squeeze().argsort()[:len(X_test)//10]
print(f"  Mean error: {abs_errors[low_unc_idx].mean():.6f}")

# Save
all_results = {
    'ensemble_mse': mse,
    'mean_uncertainty': uncertainty.mean().item(),
    'uncertainty_error_correlation': correlation,
    'expert_losses': results['losses']
}

save_run(all_results, 'section2_4',
         models=results['models'],
         epochs=args.epochs, n_experts=args.n_experts, device=str(device))

print("\n" + "="*60)
print("Section 2.4 Complete")
print("="*60)
