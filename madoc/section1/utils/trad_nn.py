import torch
import torch.nn as nn
import numpy as np

# ============= SIREN =============
class Sine(nn.Module):
    """Sine activation with frequency parameter w0"""
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class SIREN(nn.Module):
    """Sinusoidal Representation Network with specialized weight initialization"""
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
        """Initialize weights according to SIREN paper (Sitzmann et al. 2020)

        First layer: uniform(-1/n, 1/n) where n is input features
        Hidden layers: uniform(-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0)
        Final layer: uniform(-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0)
        All biases: zero initialization
        """
        with torch.no_grad():
            # First layer - special initialization for first layer with sine activation
            self.net[0].weight.uniform_(-1 / self.net[0].in_features,
                                         1 / self.net[0].in_features)
            if self.net[0].bias is not None:
                nn.init.zeros_(self.net[0].bias)

            # Hidden layers (every 2nd index starting from 2 is a Linear layer)
            for i in range(2, len(self.net)-1, 2):
                # Uniform initialization scaled by omega_0 = 30.0
                bound = np.sqrt(6 / self.net[i].in_features) / 30.0
                self.net[i].weight.uniform_(-bound, bound)
                if self.net[i].bias is not None:
                    nn.init.zeros_(self.net[i].bias)

            # Final layer (outermost linear layer)
            bound = np.sqrt(6 / self.net[-1].in_features) / 30.0
            self.net[-1].weight.uniform_(-bound, bound)
            if self.net[-1].bias is not None:
                nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x):
        return self.net(x)

# ============= Standard MLP =============
class MLP(nn.Module):
    """Standard multi-layer perceptron with configurable activation function"""
    def __init__(self, in_features=1, width=5, depth=2, activation='tanh'):
        super().__init__()
        self.activation = activation
        self.depth = depth  # Store depth for initialization
        layers = []

        # Input layer
        layers.append(nn.Linear(in_features, width))

        # Add activation after input layer
        if activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'silu':
            layers.append(nn.SiLU())

        # Hidden layers (depth - 2 means layers between input and output)
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'silu':
                layers.append(nn.SiLU())

        # Output layer (no activation after output)
        layers.append(nn.Linear(width, 1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights: He for ReLU/SiLU, Xavier for tanh

        Uses conservative initialization with smaller gain for deeper networks
        to prevent gradient explosion during initial forward passes.
        """
        # Scale down initialization for deeper networks
        gain = 1.0 / max(1.0, (self.depth - 1) / 2.0)

        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                if self.activation == 'relu' or self.activation == 'silu':
                    # He initialization for ReLU and SiLU with depth-dependent gain
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    module.weight.data *= gain
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif self.activation == 'tanh':
                    # Xavier initialization for tanh with depth-dependent gain
                    nn.init.xavier_normal_(module.weight, gain=gain)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)