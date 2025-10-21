#!/bin/bash
# Wrapper script to submit Section 1.1 (Function Approximation) job
# Usage: bash submit_section1_1.sh [EPOCHS]
# Example: bash submit_section1_1.sh 100

EPOCHS=${1:-100}  # Default: 100 epochs

echo "Submitting Section 1.1: Function Approximation"
echo "Epochs: $EPOCHS"
echo ""

qsub -v SCRIPT=section1_1.py,EPOCHS=$EPOCHS run_section1.pbs
