#!/bin/bash
# =============================================================================
# test_section2_3_parallel.sh - Local testing script for parallelized Merge_KAN
# =============================================================================
#
# This script tests the parallelized Merge_KAN implementation locally without
# PBS, by running a small-scale experiment sequentially.
#
# USAGE:
#   ./test_section2_3_parallel.sh [DIM]
#
# EXAMPLES:
#   ./test_section2_3_parallel.sh 2    # Test with 2D (default)
#   ./test_section2_3_parallel.sh 4    # Test with 4D
#
# =============================================================================

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse dimension argument
DIM=${1:-2}

if [[ ! "$DIM" =~ ^(2|4|10)$ ]]; then
    echo "ERROR: Dimension must be 2, 4, or 10 (got: $DIM)"
    echo "Usage: $0 [DIM]"
    exit 1
fi

# Test parameters (small scale for quick testing)
N_SEEDS=2           # Only 2 seeds instead of 5
EXPERT_EPOCHS=10    # Only 10 epochs instead of 1000
MERGED_EPOCHS=5     # Only 5 steps per grid instead of 200
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Output directories
EXPERT_DIR="${SCRIPT_DIR}/test_experts_${DIM}d_${TIMESTAMP}"
OUTPUT_DIR="${SCRIPT_DIR}/section2/test_results"

echo ""
echo "=========================================="
echo "Local Test: Parallelized Merge_KAN"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Dimension:     ${DIM}D"
echo "  N_SEEDS:       ${N_SEEDS} (reduced for testing)"
echo "  Expert Epochs: ${EXPERT_EPOCHS} (reduced for testing)"
echo "  Merged Epochs: ${MERGED_EPOCHS} (reduced for testing)"
echo "  Expert Dir:    ${EXPERT_DIR}"
echo "  Output Dir:    ${OUTPUT_DIR}"
echo ""

# Create directories
mkdir -p "${EXPERT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# Phase 1: Train Experts (Sequential simulation of parallel)
# =============================================================================

echo "=========================================="
echo "Phase 1: Training Experts (Sequential)"
echo "=========================================="
echo ""

# Calculate number of experts (3 * N_SEEDS)
N_EXPERTS=$((3 * N_SEEDS))

echo "Training ${N_EXPERTS} experts sequentially..."
echo ""

for i in $(seq 0 $((N_EXPERTS - 1))); do
    echo "--------------------------------------"
    echo "Training Expert ${i}/${N_EXPERTS}..."
    echo "--------------------------------------"

    python3 "${SCRIPT_DIR}/section2/section2_3_train_expert.py" \
        --index ${i} \
        --dim ${DIM} \
        --n-seeds ${N_SEEDS} \
        --epochs ${EXPERT_EPOCHS} \
        --output-dir "${EXPERT_DIR}" \
        2>&1 | tee "${EXPERT_DIR}/expert_${i}.log"

    if [[ $? -eq 0 ]]; then
        echo "✓ Expert ${i} trained successfully"
        touch "${EXPERT_DIR}/expert_${i}.success"
    else
        echo "✗ Expert ${i} failed!"
        exit 1
    fi

    echo ""
done

echo "Phase 1 Complete: All ${N_EXPERTS} experts trained"
echo ""

# =============================================================================
# Phase 2: Merge and Train
# =============================================================================

echo "=========================================="
echo "Phase 2: Merge and Train"
echo "=========================================="
echo ""

python3 "${SCRIPT_DIR}/section2/section2_3_merge.py" \
    --dim ${DIM} \
    --expert-dir "${EXPERT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --merged-epochs ${MERGED_EPOCHS} \
    --grids 3 5 10 \
    2>&1 | tee "${OUTPUT_DIR}/section2_3_merge_${DIM}d_test.log"

if [[ $? -eq 0 ]]; then
    echo ""
    echo "✓ Phase 2 completed successfully"
else
    echo ""
    echo "✗ Phase 2 failed!"
    exit 1
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=========================================="
echo "Local Test Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Expert Models:  ${EXPERT_DIR}"
echo "  Final Results:  ${OUTPUT_DIR}/section2_3_${DIM}d"
echo ""
echo "Trained Experts:"
ls -1 "${EXPERT_DIR}"/expert_*.pkl | head -n 5
echo "  ... (total: $(ls -1 ${EXPERT_DIR}/expert_*.pkl | wc -l))"
echo ""
echo "Result Files:"
ls -1 "${OUTPUT_DIR}"/section2_3_${DIM}d/*.csv 2>/dev/null || echo "  (CSV files in results directory)"
echo ""
echo "Next Steps:"
echo "  1. Review logs: cat ${EXPERT_DIR}/expert_0.log"
echo "  2. Check results: cat ${OUTPUT_DIR}/section2_3_${DIM}d/summary.csv"
echo "  3. Run full experiment on PBS: ./submit_section2_3.sh --dim ${DIM}"
echo ""

exit 0
