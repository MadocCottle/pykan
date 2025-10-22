Yes, it is absolutely possible to directly access the heuristic used in model.prune() to perform different operations on nodes that would have been selected for pruning. The pruning heuristic is based on attribution scores that are computed and stored as model attributes. Here's what you need to know:
Key Components
Attribution Scores: The model.attribute() method (MultKAN.py:1913) computes attribution scores using a backward propagation algorithm that measures each node's and edge's contribution to the output.
Stored Attributes: After calling model.attribute(), the following are available:
model.node_scores - List of average node attribution scores per layer
model.edge_scores - List of average edge attribution scores per layer
model.node_scores_all - Full node scores (not averaged)
model.edge_scores_all - Full edge scores (not averaged)
model.subnode_scores - Subnode attribution scores
How prune() Uses These:
At MultKAN.py:1655-1656, prune_node() calls self.attribute() and then compares self.node_scores[i+1] > threshold
At MultKAN.py:1777, prune_edge() uses self.edge_scores[i] > threshold
How to Access and Use the Heuristic
# Step 1: Ensure activations are cached
if model.acts is None:
    model.get_act()

# Step 2: Compute attribution scores
model.attribute()

# Step 3: Access the scores directly
# For nodes at layer i:
node_scores_layer_i = model.node_scores[i]  # Shape: (num_nodes,)

# For edges between layer i-1 and i:
edge_scores_layer_i = model.edge_scores[i-1]  # Shape: (num_outputs, num_inputs)

# Step 4: Apply your custom operation instead of pruning
threshold = 1e-2
nodes_to_operate_on = node_scores_layer_i < threshold  # Boolean mask
# Now perform your custom operation on these nodes

#### Applying densification:

Answer: Selective Grid Densification Based on Attribution Scores
Yes, it is absolutely possible to selectively densify grids for specific nodes/edges based on attribution scores! Here's how pykan's densification works and how you can adapt it:
How PyKAN Uses Grid Densification
1. Global Refinement (MultKAN.py:437-492)
PyKAN's default model.refine(new_grid) method:
Creates an entirely new model with a denser grid for ALL activation functions
Uses initialize_grid_from_parent() to transfer learned functions from coarse to fine grid
Applies to all layers uniformly - no selectivity
2. Grid Structure (KANLayer.py:98-100)
Each KANLayer has:
self.grid: Shape (in_dim, num_intervals + 1 + 2*k)
One grid per input dimension (each input neuron gets its own grid)
self.coef: Shape (in_dim, out_dim, num_intervals + k) - B-spline coefficients
This means grids are already per-neuron, not global!
3. Grid Update Process (KANLayer.py:169-217)
The update_grid_from_samples() method:
Sorts input samples to find data distribution
Creates adaptive grid based on percentile distribution
Evaluates current spline on old grid
Refits spline coefficients to new grid (preserving learned function)
Strategy for Selective Densification
Since grids are already per-input-neuron in each layer, you can selectively refine specific neurons! Here's the approach:
Approach 1: Selective Per-Input-Neuron Densification
# Step 1: Get attribution scores
if model.acts is None:
    model.get_act(data)
model.attribute()

# Step 2: For each layer, identify high-attribution neurons
for layer_idx in range(model.depth):
    layer = model.act_fun[layer_idx]
    
    # Get node scores for this layer's inputs
    # For edges, you might want edge_scores instead
    node_scores = model.node_scores[layer_idx]
    
    # Identify which input neurons to densify (high attribution = keep dense)
    threshold = 1e-2
    neurons_to_densify = node_scores > threshold
    
    # Step 3: Selectively update grid for high-importance neurons only
    for input_neuron_idx in range(layer.in_dim):
        if neurons_to_densify[input_neuron_idx]:
            # Densify this specific neuron's grid
            # Need to create new finer grid for this input dimension
            pass  # Implementation below
Approach 2: Edge-Level Selective Densification
Since coef has shape (in_dim, out_dim, num_coef), you could theoretically have different grid densities per edge, but this is not directly supported - all edges from the same input neuron share the same grid.
Reusable Components from PyKAN
What You Can Reuse:
Grid Generation Logic (KANLayer.py:197-204)
def get_grid(num_interval, x_pos, grid_eps):
    ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
    grid_adaptive = x_pos[ids, :].permute(1,0)
    margin = 0.00
    h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]] + 2 * margin)/num_interval
    grid_uniform = grid_adaptive[:,[0]] - margin + h * torch.arange(num_interval+1,)[None, :].to(x.device)
    grid = grid_eps * grid_uniform + (1 - grid_eps) * grid_adaptive
    return grid
Function Preservation (KANLayer.py:214-217)
# Evaluate old function
y_eval = coef2curve(x_pos, old_grid, old_coef, k)
# Refit to new grid
new_coef = curve2coef(x_pos, y_eval, new_grid, k)
Grid Extension (spline.py:134-143)
from kan.spline import extend_grid
grid = extend_grid(grid, k_extend=k)
Implementation Strategy
Option A: Modify Grid In-Place (Per Input Neuron)
def selective_densify_by_attribution(model, data, threshold=1e-2, new_grid_size=10):
    """
    Densify grids only for neurons with attribution > threshold.
    """
    # Get activations and attribution
    if model.acts is None:
        model.get_act(data)
    model.attribute()
    
    for layer_idx in range(model.depth):
        layer = model.act_fun[layer_idx]
        acts = model.acts[layer_idx]  # Activations for this layer
        node_scores = model.node_scores[layer_idx]
        
        # Process each input dimension separately
        for in_idx in range(layer.in_dim):
            if node_scores[in_idx] > threshold:
                # Extract current grid and coef for this neuron
                old_grid = layer.grid[in_idx:in_idx+1, :]  # Shape (1, old_num)
                old_coef = layer.coef[in_idx:in_idx+1, :, :]  # Shape (1, out_dim, old_num)
                
                # Get samples for this input dimension
                x_samples = acts[:, in_idx:in_idx+1]  # Shape (batch, 1)
                x_pos = torch.sort(x_samples, dim=0)[0]
                
                # Evaluate current function
                y_eval = coef2curve(x_pos, old_grid, old_coef, layer.k)
                
                # Create new denser grid
                num_interval = new_grid_size
                batch = x_samples.shape[0]
                ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
                grid_adaptive = x_pos[ids, :].permute(1,0)
                h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]])/num_interval
                grid_uniform = grid_adaptive[:,[0]] + h * torch.arange(num_interval+1,)[None, :].to(x_samples.device)
                new_grid = layer.grid_eps * grid_uniform + (1 - layer.grid_eps) * grid_adaptive
                new_grid = extend_grid(new_grid, k_extend=layer.k)
                
                # Refit coefficients to new grid
                new_coef = curve2coef(x_pos, y_eval, new_grid, layer.k)
                
                # Update this neuron's grid and coef
                layer.grid.data[in_idx:in_idx+1, :] = new_grid
                layer.coef.data[in_idx:in_idx+1, :, :] = new_coef
Limitation: self.grid must have uniform shape (in_dim, grid_size), so all input neurons must have the same number of grid points. You'd need to use the maximum grid size and pad/interpolate