#!/bin/bash
# setup.sh - Environment setup for madoc experiments on NCI Gadi
#
# This script creates a Python virtual environment and installs all required
# dependencies for running madoc experiments.
#
# Usage:
#   bash setup.sh
#
# Requirements:
#   - Python 3.10+ (load with: module load python3/3.10.4 on Gadi)
#   - Internet access (for pip install)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "=========================================="
echo "PyKAN Madoc Environment Setup"
echo "=========================================="
echo ""

# Check if running on Gadi (multiple detection methods)
ON_GADI=false
if [[ -f /etc/nci-environment ]]; then
    echo "Detected NCI Gadi environment (via /etc/nci-environment)"
    ON_GADI=true
elif [[ $(hostname) =~ gadi ]]; then
    echo "Detected NCI Gadi environment (via hostname)"
    ON_GADI=true
elif [[ -n "${PBS_O_WORKDIR}" ]] || [[ -n "${PBS_JOBID}" ]]; then
    echo "Detected NCI Gadi environment (via PBS variables)"
    ON_GADI=true
elif command -v module >/dev/null 2>&1 && module avail python3 2>&1 | grep -q "python3/3.10"; then
    echo "Detected environment with module system (assuming Gadi/HPC)"
    ON_GADI=true
fi

if [[ "$ON_GADI" == false ]]; then
    echo "Running on local/non-Gadi environment"
fi
echo ""

# Load Python module if on Gadi or if module command is available
if [[ "$ON_GADI" == true ]] || command -v module >/dev/null 2>&1; then
    echo "Attempting to load Python module..."

    # Try to load the module command if not already available
    if ! command -v module >/dev/null 2>&1; then
        # Common locations for module initialization
        for init_script in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
            if [[ -f "$init_script" ]]; then
                echo "Initializing module system from $init_script"
                source "$init_script"
                break
            fi
        done
    fi

    if command -v module >/dev/null 2>&1; then
        # Try to load Python 3.10.4 or newer
        if module load python3/3.10.4 2>/dev/null; then
            echo "✓ Loaded python3/3.10.4 module"
        elif module load python3/3.10 2>/dev/null; then
            echo "✓ Loaded python3/3.10 module"
        elif module load python3/3.11 2>/dev/null; then
            echo "✓ Loaded python3/3.11 module"
        elif module load python3/3.12 2>/dev/null; then
            echo "✓ Loaded python3/3.12 module"
        else
            echo "WARNING: Could not load Python 3.10+ module automatically"
            echo "Available Python modules:"
            module avail python3 2>&1 | head -20
            echo ""
            echo "Please load a Python 3.10+ module manually:"
            echo "  module load python3/3.10.4"
            echo "Then re-run this script."
            exit 1
        fi
        echo ""
    else
        echo "WARNING: 'module' command not found"
        echo "Proceeding with system Python..."
        echo ""
    fi
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Using Python version: $PYTHON_VERSION"

# Verify Python version is 3.10+
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 8 ]]; then
    echo ""
    echo "=========================================="
    echo "ERROR: Python version too old"
    echo "=========================================="
    echo "Current version: $PYTHON_VERSION"
    echo "Required: Python 3.8+ (3.10+ recommended)"
    echo ""
    echo "This version is too old to install PyTorch 2.2.2 and other dependencies."
    echo ""
    if command -v module >/dev/null 2>&1; then
        echo "On Gadi/HPC systems, load a newer Python module:"
        echo "  module load python3/3.10.4"
        echo "  bash setup.sh"
        echo ""
        echo "Available Python 3 modules:"
        module avail python3 2>&1 | grep -E "python3/3\.(8|9|10|11|12)" | head -10
    else
        echo "Please install Python 3.10 or newer and try again."
    fi
    echo ""
    exit 1
elif [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 10 ]]; then
    echo "WARNING: Python 3.10+ recommended for best compatibility, you have $PYTHON_VERSION"
    echo "         Some features may not work correctly with Python $PYTHON_VERSION"
fi
echo ""

# Create virtual environment if it doesn't exist
if [[ -d "$VENV_DIR" ]]; then
    echo "Virtual environment already exists at: $VENV_DIR"
    read -p "Recreate virtual environment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Keeping existing virtual environment"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created successfully"
else
    echo "Using existing virtual environment"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"
echo "Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install requirements
if [[ -f "${SCRIPT_DIR}/requirements.txt" ]]; then
    echo "Installing packages from requirements.txt..."
    echo "This may take several minutes..."
    pip install -r "${SCRIPT_DIR}/requirements.txt"
    echo ""
    echo "Packages installed successfully"
else
    echo "ERROR: requirements.txt not found at ${SCRIPT_DIR}/requirements.txt"
    exit 1
fi
echo ""

# Verify PyKAN can be imported
echo "Verifying PyKAN import..."
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('${SCRIPT_DIR}').parent))
try:
    from kan import *
    print('✓ PyKAN imported successfully')
except ImportError as e:
    print('✗ Failed to import PyKAN:', e)
    sys.exit(1)
"

if [[ $? -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "Setup completed successfully!"
    echo "=========================================="
    echo ""
    echo "Virtual environment location: $VENV_DIR"
    echo ""
    echo "To activate the environment manually:"
    echo "  source ${VENV_DIR}/bin/activate"
    echo ""
    echo "To run experiments:"
    echo "  1. Activate the environment (see above)"
    echo "  2. Run a section script, e.g.:"
    echo "     python3 section1/section1_1.py --epochs 100"
    echo ""
    echo "To submit a job on Gadi:"
    echo "  qsub -v SECTION=section1_1,EPOCHS=100 run_experiment.qsub"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Setup completed with warnings"
    echo "=========================================="
    echo ""
    echo "Virtual environment created but PyKAN import verification failed."
    echo "This may be normal if pykan dependencies are not yet installed."
    echo "Please check the parent directory contains the pykan package."
    exit 1
fi
