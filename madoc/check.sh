#!/bin/bash

# check.sh - Quick validation script to test all sections with minimal epochs
# This script runs all section scripts with minimal epochs to verify they work
# Usage: ./check.sh [epochs] [--section1_only|--section2_only]
# Default epochs: 5
# Default: runs all sections

set -e  # Exit on any error

# Parse command line arguments
EPOCHS=${1:-5}  # Default to 5 if no argument provided
SECTION1_ONLY=false
SECTION2_ONLY=false

# Check for section flags
for arg in "$@"; do
    case $arg in
        --section1_only)
            SECTION1_ONLY=true
            ;;
        --section2_only)
            SECTION2_ONLY=true
            ;;
    esac
done

# Determine which sections to run
if [ "$SECTION1_ONLY" = true ] && [ "$SECTION2_ONLY" = true ]; then
    echo "Error: Cannot specify both --section1_only and --section2_only"
    exit 1
fi

SECTIONS_MSG="all sections"
if [ "$SECTION1_ONLY" = true ]; then
    SECTIONS_MSG="Section 1 only"
elif [ "$SECTION2_ONLY" = true ]; then
    SECTIONS_MSG="Section 2 only"
fi

echo "=========================================="
echo "Starting Quick Validation Check"
echo "Using $EPOCHS epochs for $SECTIONS_MSG"
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

# Section 1 scripts
if [ "$SECTION2_ONLY" = false ]; then
    # Section 1.1: Function Approximation
    # Uses --epochs for total budget and --steps_per_grid for KAN grid steps
    echo "=== Section 1.1: Function Approximation ==="
    "$PYTHON" "$SCRIPT_DIR/section1/section1_1.py" --epochs "$EPOCHS" --steps_per_grid 5
    echo "✓ Section 1.1 completed"
    echo ""

    # Section 1.2: 1D Poisson PDE
    echo "=== Section 1.2: 1D Poisson PDE ==="
    "$PYTHON" "$SCRIPT_DIR/section1/section1_2.py" --epochs "$EPOCHS"
    echo "✓ Section 1.2 completed"
    echo ""

    # Section 1.3
    echo "=== Section 1.3 ==="
    "$PYTHON" "$SCRIPT_DIR/section1/section1_3.py" --epochs "$EPOCHS"
    echo "✓ Section 1.3 completed"
    echo ""
fi

# Section 2 scripts
if [ "$SECTION1_ONLY" = false ]; then
    # Section 2.1: Optimizer Comparison
    echo "=== Section 2.1: Optimizer Comparison ==="
    "$PYTHON" "$SCRIPT_DIR/section2/section2_1.py" --epochs "$EPOCHS"
    echo "✓ Section 2.1 completed"
    echo ""

    # Section 2.2: 2D experiments
    echo "=== Section 2.2 ==="
    "$PYTHON" "$SCRIPT_DIR/section2/section2_2.py" --epochs "$EPOCHS"
    echo "✓ Section 2.2 completed"
    echo ""

    # Section 2.3: Final experiments
    echo "=== Section 2.3 ==="
    "$PYTHON" "$SCRIPT_DIR/section2/section2_3.py" --epochs "$EPOCHS"
    echo "✓ Section 2.3 completed"
    echo ""
fi

echo "=========================================="
echo "All requested sections validated successfully! ✓"
echo "Completed with $EPOCHS epochs"
echo "=========================================="