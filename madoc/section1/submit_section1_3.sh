#!/bin/bash
# Wrapper script to submit Section 1.3 (2D Poisson PDE) job
# Usage: bash submit_section1_3.sh [EPOCHS]
# Example: bash submit_section1_3.sh 200

EPOCHS=${1:-200}  # Default: 200 epochs

echo "Submitting Section 1.3: 2D Poisson PDE"
echo "Epochs: $EPOCHS"
echo ""

qsub -v SCRIPT=section1_3.py,EPOCHS=$EPOCHS run_section1.pbs
