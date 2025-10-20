from kan import *

# ============= Section 1.1: Function Approximation =============

def sinusoid_1d(freq):
    return lambda x: torch.sin(2 * torch.pi * freq * x)

# Piecewise function
f_piecewise = lambda x: torch.where(x < 0.3, -0.5, torch.where(x < 0.6, 0.3, torch.where(x < 0.8, 1.0, -0.2)))

# Sawtooth wave
f_sawtooth = lambda x: (x % 0.25) / 0.25

# Polynomial
f_polynomial = lambda x: x**3 - 2*x**2 + x


# 2D Sinusoid products
f_2d_sin_1x1 = lambda x: torch.sin(2*torch.pi*1*x[:,[0]]) * torch.sin(2*torch.pi*1*x[:,[1]])
f_2d_sin_2x2 = lambda x: torch.sin(2*torch.pi*2*x[:,[0]]) * torch.sin(2*torch.pi*2*x[:,[1]])
f_2d_sin_3x3 = lambda x: torch.sin(2*torch.pi*3*x[:,[0]]) * torch.sin(2*torch.pi*3*x[:,[1]])

def sinusoid_2d(freq_x, freq_y):
    return lambda x: torch.sin(2*torch.pi*freq_x*x[:,[0]]) * torch.sin(2*torch.pi*freq_y*x[:,[1]])

# Gaussian bumps (example with 3 bumps)
f_gaussian_bumps = lambda x: (torch.exp(-((x[:,[0]]-0.3)**2 + (x[:,[1]]-0.3)**2)/0.02) + 
                               torch.exp(-((x[:,[0]]-0.7)**2 + (x[:,[1]]-0.7)**2)/0.02) +
                               torch.exp(-((x[:,[0]]-0.5)**2 + (x[:,[1]]-0.5)**2)/0.01))

# ============= Section 1.2: 1D Poisson PDE (forcing functions) =============
# Standard sin forcing
f_poisson_1d_sin = lambda x: (torch.pi**2) * torch.sin(torch.pi * x)

# Polynomial forcing
f_poisson_1d_poly = lambda x: 2 * torch.ones_like(x)

# High frequency sin forcing
f_poisson_1d_highfreq = lambda x: 16 * (torch.pi**2) * torch.sin(4 * torch.pi * x)

sec1_2 = [f_poisson_1d_sin, f_poisson_1d_poly, f_poisson_1d_highfreq]

# ============= #Section 1.3: 2D Poisson PDE (forcing functions) =============
#Section 1.3: 2D Poisson PDE (forcing functions)

# 2D sin forcing
f_poisson_2d_sin = lambda x: 2 * (torch.pi**2) * torch.sin(torch.pi*x[:,[0]]) * torch.sin(torch.pi*x[:,[1]])

# 2D polynomial forcing
f_poisson_2d_poly = lambda x: 2*x[:,[1]]*(1-x[:,[1]]) + 2*x[:,[0]]*(1-x[:,[0]])

# 2D high frequency forcing
f_poisson_2d_highfreq = lambda x: 32 * (torch.pi**2) * torch.sin(4*torch.pi*x[:,[0]]) * torch.sin(4*torch.pi*x[:,[1]])

# Special data spec forcing
f_poisson_2d_spec = lambda x: -(torch.pi**2) * (1 + 4*x[:,[1]]**2) * torch.sin(torch.pi*x[:,[0]]) * torch.sin(torch.pi*x[:,[1]]**2) + 2*torch.pi*torch.sin(torch.pi*x[:,[0]])*torch.cos(torch.pi*x[:,[1]]**2)