#!/bin/bash
# Setup script for Section 1 experiments on NCI Gadi
# This script sets up the Python environment and installs dependencies

set -e  # Exit on error

echo "================================================"
echo "Setting up Section 1 environment on Gadi"
echo "================================================"

# Load Python module (adjust version as needed)
module purge
module load python3/3.10.4

echo "Python version:"
python3 --version

# Create virtual environment if it doesn't exist
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
else
    echo ""
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements from parent madoc directory
echo ""
echo "Installing dependencies from requirements.txt..."
cd ..
pip install -r requirements.txt
cd section1

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "To activate the environment manually, run:"
echo "  module load python3/3.10.4"
echo "  source .venv/bin/activate"
echo ""
echo "To run experiments, use the PBS script:"
echo "  qsub run_section1.pbs -v SCRIPT=section1_1.py,EPOCHS=100"
echo ""
