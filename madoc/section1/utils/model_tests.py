"""Model training and testing utilities for Section 1 experiments"""
from kan import *
from . import trad_nn as tnn
from .metrics import dense_mse_error_from_dataset
import time

def train_model(model, dataset, epochs, device, true_function=None, compute_dense_mse=False):
    """Train any model using LBFGS optimizer

    Args:
        model: Model to train
        dataset: Dataset dictionary with train/test splits
        epochs: Number of training epochs
        device: Device to train on
        true_function: Ground truth function for dense MSE calculation (optional)
        compute_dense_mse: Whether to compute dense MSE at each epoch (default: False)

    Returns:
        train_losses, test_losses, total_time, time_per_epoch, dense_mse_errors (if enabled)
    """
    optimizer = torch.optim.LBFGS(model.parameters())
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []
    dense_mse_errors = [] if compute_dense_mse else None

    start_time = time.time()

    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            train_pred = model(dataset['train_input'])
            loss = criterion(train_pred, dataset['train_label'])
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            train_pred = model(dataset['train_input'])
            test_pred = model(dataset['test_input'])
            train_loss = criterion(train_pred, dataset['train_label']).item()
            test_loss = criterion(test_pred, dataset['test_label']).item()
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            # Compute dense MSE if requested
            if compute_dense_mse and true_function is not None:
                dense_mse = dense_mse_error_from_dataset(model, dataset, true_function,
                                                         num_samples=10000, device=device)
                dense_mse_errors.append(dense_mse)

    total_time = time.time() - start_time
    time_per_epoch = total_time / epochs if epochs > 0 else 0

    if compute_dense_mse:
        return train_losses, test_losses, total_time, time_per_epoch, dense_mse_errors
    else:
        return train_losses, test_losses, total_time, time_per_epoch

def run_mlp_tests(datasets, depths, activations, epochs, device, true_functions=None, compute_dense_mse=False):
    print("mlp")
    results = {}
    for i, dataset in enumerate(datasets):
        results[i] = {}
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions[i] if true_functions else None
        for d in depths:
            results[i][d] = {}
            for act in activations:
                model = tnn.MLP(in_features=n_var, width=5, depth=d, activation=act).to(device)
                train_result = train_model(model, dataset, epochs, device, true_func, compute_dense_mse)

                if compute_dense_mse and true_func:
                    train_loss, test_loss, total_time, time_per_epoch, dense_mse = train_result
                    results[i][d][act] = {
                        'train': train_loss,
                        'test': test_loss,
                        'dense_mse': dense_mse,
                        'total_time': total_time,
                        'time_per_epoch': time_per_epoch
                    }
                else:
                    train_loss, test_loss, total_time, time_per_epoch = train_result
                    results[i][d][act] = {
                        'train': train_loss,
                        'test': test_loss,
                        'total_time': total_time,
                        'time_per_epoch': time_per_epoch
                    }
                print(f"  Dataset {i}, depth {d}, {act}: {total_time:.2f}s total, {time_per_epoch:.3f}s/epoch")
    return results

def run_siren_tests(datasets, depths, epochs, device, true_functions=None, compute_dense_mse=False):
    print("siren")
    results = {}
    for i, dataset in enumerate(datasets):
        results[i] = {}
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions[i] if true_functions else None
        for d in depths:
            model = tnn.SIREN(in_features=n_var, hidden_features=5, hidden_layers=d-2, out_features=1).to(device)
            train_result = train_model(model, dataset, epochs, device, true_func, compute_dense_mse)

            if compute_dense_mse and true_func:
                train_loss, test_loss, total_time, time_per_epoch, dense_mse = train_result
                results[i][d] = {
                    'train': train_loss,
                    'test': test_loss,
                    'dense_mse': dense_mse,
                    'total_time': total_time,
                    'time_per_epoch': time_per_epoch
                }
            else:
                train_loss, test_loss, total_time, time_per_epoch = train_result
                results[i][d] = {
                    'train': train_loss,
                    'test': test_loss,
                    'total_time': total_time,
                    'time_per_epoch': time_per_epoch
                }
            print(f"  Dataset {i}, depth {d}: {total_time:.2f}s total, {time_per_epoch:.3f}s/epoch")
    return results

def run_kan_grid_tests(datasets, grids, epochs, device, prune=False, true_functions=None, compute_dense_mse=False):
    print("kan_pruning" if prune else "kan")
    results = {}
    models = {}
    pruned_models = {} if prune else None

    for i, dataset in enumerate(datasets):
        train_losses = []
        test_losses = []
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions[i] if true_functions else None

        dataset_start_time = time.time()
        grid_times = []

        for j, grid_size in enumerate(grids):
            grid_start_time = time.time()
            if j == 0:
                model = KAN(width=[n_var, 5, 1], grid=grid_size, k=3, seed=1, device=device)
            else:
                model = model.refine(grid_size)
            train_results = model.fit(dataset, opt="LBFGS", steps=epochs, log=1)
            train_losses += train_results['train_loss']
            test_losses += train_results['test_loss']

            grid_time = time.time() - grid_start_time
            grid_times.append(grid_time)
            print(f"  Dataset {i}, grid {grid_size}: {grid_time:.2f}s total, {grid_time/epochs:.3f}s/epoch")

        total_dataset_time = time.time() - dataset_start_time
        models[i] = model

        if prune:
            prune_start_time = time.time()
            model_pruned = model.prune(node_th=1e-2, edge_th=3e-2)
            pruned_models[i] = model_pruned
            prune_time = time.time() - prune_start_time
            with torch.no_grad():
                train_loss_pruned = nn.MSELoss()(model_pruned(dataset['train_input']), dataset['train_label']).item()
                test_loss_pruned = nn.MSELoss()(model_pruned(dataset['test_input']), dataset['test_label']).item()
            print(f"  Dataset {i}, pruning: {prune_time:.2f}s")

        results[i] = {}
        for j, grid_size in enumerate(grids):
            idx = (j + 1) * epochs - 1
            result_dict = {
                'train': train_losses[idx],
                'test': test_losses[idx],
                'total_time': grid_times[j],
                'time_per_epoch': grid_times[j] / epochs
            }
            # Compute final dense MSE for this grid if requested
            if compute_dense_mse and true_func:
                # Store the model state and compute dense MSE
                # Note: we're computing it after all training, so we only get final dense MSE per grid
                # For per-epoch tracking, KAN would need to be modified more extensively
                dense_mse_final = dense_mse_error_from_dataset(model, dataset, true_func,
                                                              num_samples=10000, device=device)
                result_dict['dense_mse_final'] = dense_mse_final
            results[i][grid_size] = result_dict

        if prune:
            prune_result = {
                'train': train_loss_pruned,
                'test': test_loss_pruned,
                'prune_time': prune_time
            }
            if compute_dense_mse and true_func:
                dense_mse_pruned = dense_mse_error_from_dataset(model_pruned, dataset, true_func,
                                                                num_samples=10000, device=device)
                prune_result['dense_mse'] = dense_mse_pruned
            results[i]['pruned'] = prune_result

        results[i]['total_dataset_time'] = total_dataset_time
        print(f"  Dataset {i} complete: {total_dataset_time:.2f}s total")

    return (results, models, pruned_models) if prune else (results, models)
