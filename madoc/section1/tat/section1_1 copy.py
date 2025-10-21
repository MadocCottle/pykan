import sys
from pathlib import Path
import json
import pickle
from datetime import datetime

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
import data_funcs as dfs
import trad_nn as tnn

epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(device)

#Section 1.1: Function Approximation
# ============= Create Datasets =============
freq = [1,2,3,4,5]
datasets = []
for f in freq:
    datasets.append(create_dataset(dfs.sinusoid_1d(f), n_var=1, train_num=1000, test_num=1000))

piece_1d = create_dataset(dfs.f_piecewise, n_var=1, train_num=1000, test_num=1000)

saw_1d = create_dataset(dfs.f_sawtooth, n_var=1, train_num=1000, test_num=1000)

poly_1d = create_dataset(dfs.f_polynomial, n_var=1, train_num=1000, test_num=1000)

pde_1d = create_dataset(dfs.f_poisson_1d_highfreq, n_var=1, train_num=1000, test_num=1000)
datasets.extend([piece_1d, saw_1d, poly_1d, pde_1d])


grids = np.array([3,5,10,20,50,100])
depths = [2, 3, 4, 5, 6]
activations = ['tanh', 'relu', 'silu']

# ============= Run training =============
# ============= Claude Complete Here =============

def train_model(model, dataset, epochs):
    """Train any model using LBFGS optimizer"""
    optimizer = torch.optim.LBFGS(model.parameters())
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            train_pred = model(dataset['train_input'])
            loss = criterion(train_pred, dataset['train_label'])
            loss.backward()
            return loss

        optimizer.step(closure)
        
        # Evaluate
        with torch.no_grad():
            train_pred = model(dataset['train_input'])
            test_pred = model(dataset['test_input'])
            train_loss = criterion(train_pred, dataset['train_label']).item()
            test_loss = criterion(test_pred, dataset['test_label']).item()
            train_losses.append(train_loss)
            test_losses.append(test_loss)

    return train_losses, test_losses

def run_mlp_tests():
    print("mlp")
    results = {}
    for i, dataset in enumerate(datasets):
        results[i] = {}
        for d in depths:
            results[i][d] = {}
            for act in activations:
                model = tnn.MLP(width=5, depth=d, activation=act).to(device)
                train_loss, test_loss = train_model(model, dataset, epochs)
                results[i][d][act] = {'train': train_loss, 'test': test_loss}
    return results

def run_siren_tests():
    print("siren")
    results = {}
    for i, dataset in enumerate(datasets):
        results[i] = {}
        for d in depths:
            model = tnn.SIREN(in_features=1, hidden_features=5, hidden_layers=d-2, out_features=1).to(device)
            train_loss, test_loss = train_model(model, dataset, epochs)
            results[i][d] = {'train': train_loss, 'test': test_loss}
    return results

def run_kan_grid_tests():
    print("kan")
    results = {}
    for i, dataset in enumerate(datasets):
        train_losses = []
        test_losses = []

        # Progressive grid refinement as shown in Example_1_function_fitting.ipynb
        for j, grid_size in enumerate(grids):
            if j == 0:
                # Initialize model with first (coarsest) grid
                model = KAN(width=[1, 5, 1], grid=grid_size, k=3, seed=1, device=device)
            else:
                # Refine the existing model to a finer grid
                model = model.refine(grid_size)

            # Train on current grid resolution
            train_results = model.fit(dataset, opt="LBFGS", steps=epochs, log=0)

            # Accumulate losses to show staircase pattern
            train_losses += train_results['train_loss']
            test_losses += train_results['test_loss']

        # Store final losses at each grid resolution
        results[i] = {}
        for j, grid_size in enumerate(grids):
            # Extract loss at the end of training for each grid size
            idx = (j + 1) * epochs - 1
            results[i][grid_size] = {
                'train': train_losses[idx],
                'test': test_losses[idx]
            }
    return results


# ============= Claude Complete ADMIN =============
def save_results(results, output_dir='sec1_results'):
    """Save experiment results to both JSON and pickle formats with timestamp"""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Convert results to JSON-serializable format
    json_results = {}
    for model_name, model_results in results.items():
        json_results[model_name] = {}
        for dataset_idx, dataset_results in model_results.items():
            json_results[model_name][str(dataset_idx)] = {}
            for param, param_results in dataset_results.items():
                if isinstance(param_results, dict):
                    # Handle nested dictionaries (e.g., MLP with depth->activation)
                    json_results[model_name][str(dataset_idx)][str(param)] = {}
                    for sub_param, values in param_results.items():
                        if isinstance(values, dict):
                            json_results[model_name][str(dataset_idx)][str(param)][str(sub_param)] = {
                                k: [float(v) if isinstance(v, (list, np.ndarray)) else float(v) for v in vals] if isinstance(vals, (list, np.ndarray)) else float(vals)
                                for k, vals in values.items()
                            }
                        else:
                            json_results[model_name][str(dataset_idx)][str(param)][str(sub_param)] = {
                                'train': float(values['train'][-1]) if isinstance(values['train'], list) else float(values['train']),
                                'test': float(values['test'][-1]) if isinstance(values['test'], list) else float(values['test'])
                            }
                else:
                    # Handle simple train/test dictionary
                    json_results[model_name][str(dataset_idx)][str(param)] = {
                        'train': float(param_results['train']) if not isinstance(param_results['train'], list) else [float(x) for x in param_results['train']],
                        'test': float(param_results['test']) if not isinstance(param_results['test'], list) else [float(x) for x in param_results['test']]
                    }

    # Save as JSON
    json_path = output_path / f'section1_1_results_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to {json_path}")

    # Save as pickle (preserves Python objects exactly)
    pickle_path = output_path / f'section1_1_results_{timestamp}.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {pickle_path}")

    # Also save metadata about the experiment
    metadata = {
        'timestamp': timestamp,
        'epochs': epochs,
        'device': str(device),
        'grids': grids.tolist(),
        'depths': depths,
        'activations': activations,
        'frequencies': freq,
        'num_datasets': len(datasets)
    }
    metadata_path = output_path / f'section1_1_metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

all_results = {'mlp': run_mlp_tests(), 'siren': run_siren_tests(), 'kan': run_kan_grid_tests()}
print(all_results)

# Save results
save_results(all_results)
