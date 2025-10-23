#!/bin/bash

# check.sh - Quick validation script to test all sections with minimal epochs
# This script runs all section scripts with minimal epochs to verify they work

set -e  # Exit on any error

echo "=========================================="
echo "Starting Quick Validation Check"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Use virtual environment Python if available
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
    echo "Using virtual environment Python: $PYTHON"
elif [ -f "$SCRIPT_DIR/../.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/../.venv/bin/python"
    echo "Using virtual environment Python: $PYTHON"
else
    PYTHON="python"
    echo "Using system Python: $PYTHON"
    echo "⚠ Warning: Virtual environment not found. This may cause import errors."
fi
echo ""

# Section 1.1: Function Approximation
# Uses --epochs for total budget and --steps_per_grid for KAN grid steps
echo "=== Section 1.1: Function Approximation ==="
"$PYTHON" "$SCRIPT_DIR/section1/section1_1.py" --epochs 5 --steps_per_grid 5
echo "✓ Section 1.1 completed"
echo ""

# Section 1.2: 1D Poisson PDE
echo "=== Section 1.2: 1D Poisson PDE ==="
"$PYTHON" "$SCRIPT_DIR/section1/section1_2.py" --epochs 5
echo "✓ Section 1.2 completed"
echo ""

# Section 1.3
echo "=== Section 1.3 ==="
"$PYTHON" "$SCRIPT_DIR/section1/section1_3.py" --epochs 5
echo "✓ Section 1.3 completed"
echo ""

# Section 2.1: Optimizer Comparison
echo "=== Section 2.1: Optimizer Comparison ==="
"$PYTHON" "$SCRIPT_DIR/section2/section2_1.py" --epochs 5
echo "✓ Section 2.1 completed"
echo ""

# Section 2.2: 2D experiments
echo "=== Section 2.2 ==="
"$PYTHON" "$SCRIPT_DIR/section2/section2_2.py" --epochs 5
echo "✓ Section 2.2 completed"
echo ""

# Section 2.3: Final experiments
echo "=== Section 2.3 ==="
"$PYTHON" "$SCRIPT_DIR/section2/section2_3.py" --epochs 5
echo "✓ Section 2.3 completed"
echo ""

echo "=========================================="
echo "All sections validated successfully! ✓"
echo "=========================================="
