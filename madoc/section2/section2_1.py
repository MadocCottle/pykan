import sys
from pathlib import Path

# Add pykan to path (parent directory of madoc)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kan import *
sys.path.insert(0, str(Path(__file__).parent.parent / 'section1'))
from utils import data_funcs as dfs
from utils import save_run, track_time, print_timing_summary, count_parameters, dense_mse_error_from_dataset
from lm_optimizer import LevenbergMarquardt
import argparse
import time

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Section 2.1: Optimizer Comparison')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
args = parser.parse_args()

epochs = args.epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(device)
print(f"Running with {epochs} epochs")

# Section 2.1: Optimizer Comparison on 2D Poisson PDE
# ============= Create Datasets =============
datasets = []
true_functions = [dfs.f_poisson_2d_sin, dfs.f_poisson_2d_poly, dfs.f_poisson_2d_highfreq, dfs.f_poisson_2d_spec]

for f in true_functions:
    datasets.append(create_dataset(f, n_var=2, train_num=1000, test_num=1000))

grids = np.array([3, 5, 10, 20, 50, 100])

print("\n" + "="*60)
print("Starting Section 2.1 Optimizer Comparison")
print("="*60 + "\n")

timers = {}


def run_kan_optimizer_tests(datasets, grids, epochs, device, optimizer_name, true_functions=None, compute_dense_mse=False):
    """Run KAN tests with specified optimizer"""
    print(f"kan_{optimizer_name.lower()}")
    results = {}
    models = {}

    for i, dataset in enumerate(datasets):
        train_losses = []
        test_losses = []
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions[i] if true_functions else None

        dataset_start_time = time.time()
        grid_times = []
        grid_param_counts = []

        for j, grid_size in enumerate(grids):
            grid_start_time = time.time()
            if j == 0:
                model = KAN(width=[n_var, 5, 1], grid=grid_size, k=3, seed=1, device=device)
            else:
                model = model.refine(grid_size)
            num_params = count_parameters(model)
            grid_param_counts.append(num_params)

            # Train with specified optimizer
            train_results = model.fit(dataset, opt=optimizer_name, steps=epochs, log=1)
            train_losses += train_results['train_loss']
            test_losses += train_results['test_loss']

            grid_time = time.time() - grid_start_time
            grid_times.append(grid_time)
            print(f"  Dataset {i}, grid {grid_size}: {grid_time:.2f}s total, {grid_time/epochs:.3f}s/epoch, {num_params} params")

        total_dataset_time = time.time() - dataset_start_time
        models[i] = model

        results[i] = {}
        for j, grid_size in enumerate(grids):
            idx = (j + 1) * epochs - 1
            result_dict = {
                'train': train_losses[idx],
                'test': test_losses[idx],
                'total_time': grid_times[j],
                'time_per_epoch': grid_times[j] / epochs,
                'num_params': grid_param_counts[j]
            }
            if compute_dense_mse and true_func:
                dense_mse_final = dense_mse_error_from_dataset(model, dataset, true_func,
                                                              num_samples=10000, device=device)
                result_dict['dense_mse_final'] = dense_mse_final
            results[i][grid_size] = result_dict

        results[i]['total_dataset_time'] = total_dataset_time
        print(f"  Dataset {i} complete: {total_dataset_time:.2f}s total")

    return results, models


def run_kan_lm_tests(datasets, grids, epochs, device, true_functions=None, compute_dense_mse=False):
    """Run KAN tests with custom LM optimizer"""
    print("kan_lm")
    results = {}
    models = {}

    for i, dataset in enumerate(datasets):
        train_losses = []
        test_losses = []
        n_var = dataset['train_input'].shape[1]
        true_func = true_functions[i] if true_functions else None

        dataset_start_time = time.time()
        grid_times = []
        grid_param_counts = []

        for j, grid_size in enumerate(grids):
            grid_start_time = time.time()
            if j == 0:
                model = KAN(width=[n_var, 5, 1], grid=grid_size, k=3, seed=1, device=device)
            else:
                model = model.refine(grid_size)
            num_params = count_parameters(model)
            grid_param_counts.append(num_params)

            # Train with LM optimizer manually
            optimizer = LevenbergMarquardt(model.parameters(), lr=1.0, damping=1e-3)
            criterion = nn.MSELoss()

            for epoch in range(epochs):
                def closure():
                    optimizer.zero_grad()
                    pred = model(dataset['train_input'])
                    loss = criterion(pred, dataset['train_label'])
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

            grid_time = time.time() - grid_start_time
            grid_times.append(grid_time)
            print(f"  Dataset {i}, grid {grid_size}: {grid_time:.2f}s total, {grid_time/epochs:.3f}s/epoch, {num_params} params")

        total_dataset_time = time.time() - dataset_start_time
        models[i] = model

        results[i] = {}
        for j, grid_size in enumerate(grids):
            start_idx = j * epochs
            end_idx = (j + 1) * epochs
            result_dict = {
                'train': train_losses[end_idx - 1],
                'test': test_losses[end_idx - 1],
                'total_time': grid_times[j],
                'time_per_epoch': grid_times[j] / epochs,
                'num_params': grid_param_counts[j]
            }
            if compute_dense_mse and true_func:
                dense_mse_final = dense_mse_error_from_dataset(model, dataset, true_func,
                                                              num_samples=10000, device=device)
                result_dict['dense_mse_final'] = dense_mse_final
            results[i][grid_size] = result_dict

        results[i]['total_dataset_time'] = total_dataset_time
        print(f"  Dataset {i} complete: {total_dataset_time:.2f}s total")

    return results, models


print("Training KANs with ADAM optimizer (with dense MSE metrics)...")
adam_results, adam_models = track_time(timers, "KAN ADAM training",
                                        run_kan_optimizer_tests,
                                        datasets, grids, epochs, device, "Adam", true_functions, True)

print("\nTraining KANs with LM optimizer (with dense MSE metrics)...")
lm_results, lm_models = track_time(timers, "KAN LM training",
                                    run_kan_lm_tests,
                                    datasets, grids, epochs, device, true_functions, True)

# Print timing summary
print_timing_summary(timers, "Section 2.1", num_datasets=len(datasets))

all_results = {'adam': adam_results, 'lm': lm_results}
print(all_results)

save_run(all_results, 'section2_1',
         models={'adam': adam_models, 'lm': lm_models},
         epochs=epochs, device=str(device), grids=grids.tolist(),
         num_datasets=len(datasets))
