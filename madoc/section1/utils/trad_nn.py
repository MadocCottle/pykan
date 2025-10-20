import torch
import torch.nn as nn
import numpy as np

# ============= SIREN =============
class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0
    
    def forward(self, x):
        return torch.sin(self.w0 * x)

class SIREN(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 outermost_linear=True, first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(Sine(first_omega_0))
        
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(Sine(hidden_omega_0))
        
        final_linear = nn.Linear(hidden_features, out_features)
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            # First layer
            self.net[0].weight.uniform_(-1 / self.net[0].in_features, 
                                         1 / self.net[0].in_features)
            # Hidden layers
            for i in range(2, len(self.net)-1, 2):
                self.net[i].weight.uniform_(-np.sqrt(6 / self.net[i].in_features) / 30.0,
                                             np.sqrt(6 / self.net[i].in_features) / 30.0)
            # Final layer
            self.net[-1].weight.uniform_(-np.sqrt(6 / self.net[-1].in_features) / 30.0,
                                          np.sqrt(6 / self.net[-1].in_features) / 30.0)
    
    def forward(self, x):
        return self.net(x)

# ============= Standard MLP =============
class MLP(nn.Module):
    def __init__(self, in_features=1, width=5, depth=2, activation='tanh'):
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(in_features, width))
        
        # Hidden layers
        for _ in range(depth - 2):
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            layers.append(nn.Linear(width, width))
        
        # Output layer
        if activation == 'tanh':
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)