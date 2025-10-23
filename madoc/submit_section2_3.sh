#!/bin/bash
# =============================================================================
# submit_section2_3.sh - Submit Parallelized Merge_KAN Experiment
# =============================================================================
#
# This script orchestrates the two-phase parallelized Merge_KAN workflow:
#   Phase 1: Train experts in parallel using PBS job array
#   Phase 2: Merge experts and train combined model (depends on Phase 1)
#
# USAGE:
#   ./submit_section2_3.sh --dim DIM [OPTIONS]
#
# REQUIRED:
#   --dim DIM          Problem dimension (2, 4, or 10)
#
# OPTIONAL:
#   --n-seeds N        Number of random seeds per config (default: 5)
#   --expert-epochs N  Training epochs per expert (default: 1000)
#   --merged-epochs N  Training steps per grid for merged model (default: 200)
#   --grids "G1,G2..." Grid schedule for merged training (default: "3,5,10,20")
#   --dry-run          Show commands without submitting
#
# EXAMPLES:
#   ./submit_section2_3.sh --dim 4
#   ./submit_section2_3.sh --dim 10 --n-seeds 3 --expert-epochs 500
#   ./submit_section2_3.sh --dim 2 --dry-run
#
# =============================================================================

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Default Parameters
# =============================================================================

DIM=""
N_SEEDS=5
EXPERT_EPOCHS=1000
MERGED_EPOCHS=200
GRIDS="3,5,10,20"
DRY_RUN=0

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --dim)
            DIM="$2"
            shift 2
            ;;
        --n-seeds)
            N_SEEDS="$2"
            shift 2
            ;;
        --expert-epochs)
            EXPERT_EPOCHS="$2"
            shift 2
            ;;
        --merged-epochs)
            MERGED_EPOCHS="$2"
            shift 2
            ;;
        --grids)
            GRIDS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --dim DIM [--n-seeds N] [--expert-epochs N] [--merged-epochs N] [--grids \"G1,G2,...\"] [--dry-run]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate Parameters
# =============================================================================

if [[ -z "$DIM" ]]; then
    echo "ERROR: --dim is required"
    echo ""
    echo "Usage: $0 --dim DIM [OPTIONS]"
    echo ""
    echo "Examples:"
    echo "  $0 --dim 4"
    echo "  $0 --dim 10 --n-seeds 3 --expert-epochs 500"
    exit 1
fi

if [[ ! "$DIM" =~ ^(2|4|10)$ ]]; then
    echo "ERROR: --dim must be 2, 4, or 10 (got: $DIM)"
    exit 1
fi

if ! [[ "$N_SEEDS" =~ ^[0-9]+$ ]] || [[ "$N_SEEDS" -le 0 ]]; then
    echo "ERROR: --n-seeds must be a positive integer (got: $N_SEEDS)"
    exit 1
fi

if ! [[ "$EXPERT_EPOCHS" =~ ^[0-9]+$ ]] || [[ "$EXPERT_EPOCHS" -le 0 ]]; then
    echo "ERROR: --expert-epochs must be a positive integer (got: $EXPERT_EPOCHS)"
    exit 1
fi

if ! [[ "$MERGED_EPOCHS" =~ ^[0-9]+$ ]] || [[ "$MERGED_EPOCHS" -le 0 ]]; then
    echo "ERROR: --merged-epochs must be a positive integer (got: $MERGED_EPOCHS)"
    exit 1
fi

# =============================================================================
# Calculate Resources
# =============================================================================

# Number of experts to train (3 * N_SEEDS)
N_EXPERTS=$((3 * N_SEEDS))
ARRAY_MAX=$((N_EXPERTS - 1))

# Timestamp for output directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Output directories
EXPERT_DIR="${SCRIPT_DIR}/experts_${DIM}d_${TIMESTAMP}"
OUTPUT_DIR="${SCRIPT_DIR}/section2/results"

# Phase 1 resources (per expert job)
# Small jobs: 1 CPU, 4GB, 2 hours should be enough for most cases
PHASE1_NCPUS=1
PHASE1_MEM=4GB
PHASE1_WALLTIME=02:00:00

# Phase 2 resources (merge and train)
# Larger job: 12 CPUs, 48GB, 4 hours for merging and training
PHASE2_NCPUS=12
PHASE2_MEM=48GB
PHASE2_WALLTIME=04:00:00

# =============================================================================
# Display Configuration
# =============================================================================

echo ""
echo "=========================================="
echo "Submitting Parallelized Merge_KAN Experiment"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Dimension:         ${DIM}D"
echo "  N_SEEDS:           ${N_SEEDS}"
echo "  Expert Epochs:     ${EXPERT_EPOCHS}"
echo "  Merged Epochs:     ${MERGED_EPOCHS}"
echo "  Grid Schedule:     ${GRIDS}"
echo ""
echo "Parallelization:"
echo "  Total Experts:     ${N_EXPERTS}"
echo "  Job Array Range:   0-${ARRAY_MAX}"
echo ""
echo "Phase 1 (Expert Training):"
echo "  Resource per job:  ${PHASE1_NCPUS} CPU, ${PHASE1_MEM}, ${PHASE1_WALLTIME}"
echo "  Peak CPUs:         ${N_EXPERTS} (all jobs in parallel)"
echo "  Expert Output:     ${EXPERT_DIR}"
echo ""
echo "Phase 2 (Merge and Train):"
echo "  Resources:         ${PHASE2_NCPUS} CPUs, ${PHASE2_MEM}, ${PHASE2_WALLTIME}"
echo "  Results Output:    ${OUTPUT_DIR}"
echo ""
echo "=========================================="
echo ""

# =============================================================================
# Submit Phase 1: Expert Training (Job Array)
# =============================================================================

PHASE1_CMD="qsub"
PHASE1_CMD="$PHASE1_CMD -J 0-${ARRAY_MAX}"
PHASE1_CMD="$PHASE1_CMD -v DIM=${DIM},N_SEEDS=${N_SEEDS},EPOCHS=${EXPERT_EPOCHS},OUTPUT_DIR=${EXPERT_DIR}"
PHASE1_CMD="$PHASE1_CMD -l ncpus=${PHASE1_NCPUS} -l mem=${PHASE1_MEM}"
PHASE1_CMD="$PHASE1_CMD -l walltime=${PHASE1_WALLTIME}"
PHASE1_CMD="$PHASE1_CMD ${SCRIPT_DIR}/section2_3_experts.qsub"

echo "Phase 1: Submitting expert training job array..."
echo ""
echo "Command:"
echo "  $PHASE1_CMD"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY RUN] Would submit Phase 1 job array"
    PHASE1_JOB_ID="12345[]"
else
    PHASE1_JOB_ID=$(eval "$PHASE1_CMD")
    PHASE1_EXIT_CODE=$?

    if [[ $PHASE1_EXIT_CODE -ne 0 ]]; then
        echo "✗ Phase 1 submission failed with exit code $PHASE1_EXIT_CODE"
        exit $PHASE1_EXIT_CODE
    fi

    echo "✓ Phase 1 job array submitted!"
    echo "  Job ID: $PHASE1_JOB_ID"
fi

echo ""

# =============================================================================
# Submit Phase 2: Merge and Train (Depends on Phase 1)
# =============================================================================

# Extract base job ID (remove [] suffix if present)
BASE_JOB_ID=$(echo "$PHASE1_JOB_ID" | sed 's/\[\]$//')

PHASE2_CMD="qsub"
PHASE2_CMD="$PHASE2_CMD -W depend=afterokarray:${BASE_JOB_ID}"
PHASE2_CMD="$PHASE2_CMD -v DIM=${DIM},EXPERT_DIR=${EXPERT_DIR},MERGED_EPOCHS=${MERGED_EPOCHS},OUTPUT_DIR=${OUTPUT_DIR},GRIDS=${GRIDS}"
PHASE2_CMD="$PHASE2_CMD -l ncpus=${PHASE2_NCPUS} -l mem=${PHASE2_MEM}"
PHASE2_CMD="$PHASE2_CMD -l walltime=${PHASE2_WALLTIME}"
PHASE2_CMD="$PHASE2_CMD ${SCRIPT_DIR}/section2_3_merge.qsub"

echo "Phase 2: Submitting merge and train job (depends on Phase 1)..."
echo ""
echo "Command:"
echo "  $PHASE2_CMD"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY RUN] Would submit Phase 2 job with dependency on ${BASE_JOB_ID}"
    PHASE2_JOB_ID="12346"
else
    PHASE2_JOB_ID=$(eval "$PHASE2_CMD")
    PHASE2_EXIT_CODE=$?

    if [[ $PHASE2_EXIT_CODE -ne 0 ]]; then
        echo "✗ Phase 2 submission failed with exit code $PHASE2_EXIT_CODE"
        exit $PHASE2_EXIT_CODE
    fi

    echo "✓ Phase 2 job submitted!"
    echo "  Job ID: $PHASE2_JOB_ID"
    echo "  Dependency: afterokarray:${BASE_JOB_ID}"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================

echo "=========================================="
echo "Submission Complete!"
echo "=========================================="
echo ""
echo "Job Summary:"
echo "  Phase 1 (Experts): $PHASE1_JOB_ID"
echo "  Phase 2 (Merge):   $PHASE2_JOB_ID"
echo ""
echo "Monitor jobs with:"
echo "  qstat -u \$USER"
echo "  qstat -f $BASE_JOB_ID  # Detailed Phase 1 info"
echo "  qstat -f $PHASE2_JOB_ID  # Detailed Phase 2 info"
echo ""
echo "View logs when complete:"
echo "  ls ${EXPERT_DIR}/*.log         # Individual expert logs"
echo "  ls ${OUTPUT_DIR}/*.log         # Merge and train log"
echo ""
echo "Expected Timeline:"
echo "  Phase 1: ~${EXPERT_EPOCHS} epochs × ~1min/100epochs ≈ $((EXPERT_EPOCHS / 100)) minutes per expert"
echo "  Phase 2: Starts automatically after all Phase 1 jobs complete"
echo "  Total:   ~$((EXPERT_EPOCHS / 100 + 30)) minutes (with parallelization)"
echo ""
echo "Results will be saved to:"
echo "  ${OUTPUT_DIR}/section2_3_${DIM}d/"
echo ""

exit 0
