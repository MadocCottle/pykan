#!/bin/bash
# Gadi Setup Script for Evolutionary KAN
# This script sets up the virtual environment and dependencies on Gadi

set -e  # Exit on error

echo "=========================================="
echo "Gadi Setup for Evolutionary KAN"
echo "=========================================="

# Configuration
PROJECT_DIR="${HOME}/KAN_Repo"
VENV_DIR="${PROJECT_DIR}/.venv"

# Load required modules on Gadi
echo "Loading Python module..."
module load python3/3.10.4

# Create project directory if it doesn't exist
echo "Creating project directory: ${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}"

# Create virtual environment
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib scikit-learn

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python3 -c "import scipy; print(f'SciPy version: {scipy.__version__}')"
python3 -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Virtual environment: ${VENV_DIR}"
echo "To activate: source ${VENV_DIR}/bin/activate"
echo ""
echo "Next steps:"
echo "1. Copy your KAN_Repo code to ${PROJECT_DIR}"
echo "2. Submit the PBS job: qsub gadi_run_evolution.pbs"
