#!/bin/bash
# =============================================================================
# submit_experiment.sh - Wrapper for PBS job submission with correct resources
# =============================================================================
#
# This script submits experiments to PBS with the correct resource allocation.
# It solves the problem that PBS directives (#PBS) cannot use bash variables.
#
# USAGE:
#   ./submit_experiment.sh SECTION EPOCHS [PROFILE]
#
# EXAMPLES:
#   ./submit_experiment.sh section1_1 100
#   ./submit_experiment.sh section1_1 500 large
#   ./submit_experiment.sh section2_1 50 section2
#
# PARAMETERS:
#   SECTION  - Which section to run (e.g., section1_1, section1_2, section2_1)
#   EPOCHS   - Number of training epochs (e.g., 100, 500, 1000)
#   PROFILE  - Resource profile (test, section1, section2, large)
#              If omitted, auto-selected based on SECTION
#
# PROFILES:
#   test     - 1 CPU,  4GB RAM,   30 minutes  (quick testing)
#   section1 - 12 CPUs, 48GB RAM,  4 hours    (section1 experiments)
#   section2 - 24 CPUs, 96GB RAM,  8 hours    (section2 experiments)
#   large    - 48 CPUs, 190GB RAM, 24 hours   (long runs, high memory)
#
# =============================================================================

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Parse Arguments
# =============================================================================

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 SECTION EPOCHS [PROFILE]"
    echo ""
    echo "Examples:"
    echo "  $0 section1_1 100"
    echo "  $0 section1_1 500 large"
    echo "  $0 section2_1 50"
    echo ""
    echo "Profiles: test, section1, section2, large"
    echo "  (auto-selected based on SECTION if not specified)"
    exit 1
fi

SECTION="$1"
EPOCHS="$2"
PROFILE="${3:-auto}"

# Validate EPOCHS is a positive integer
if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || [[ "$EPOCHS" -le 0 ]]; then
    echo "ERROR: EPOCHS must be a positive integer, got: $EPOCHS"
    exit 1
fi

# =============================================================================
# Auto-select Profile
# =============================================================================

if [[ "$PROFILE" == "auto" ]]; then
    if [[ "$SECTION" =~ ^section1 ]]; then
        PROFILE="section1"
        echo "Auto-selected profile: section1 (based on section name)"
    elif [[ "$SECTION" =~ ^section2 ]]; then
        PROFILE="section2"
        echo "Auto-selected profile: section2 (based on section name)"
    else
        PROFILE="section1"
        echo "Auto-selected profile: section1 (default)"
    fi
fi

# =============================================================================
# Map Profile to Resources
# =============================================================================

case "$PROFILE" in
    test)
        NCPUS=1
        MEM=4GB
        WALLTIME=00:30:00
        ;;
    section1)
        NCPUS=12
        MEM=48GB
        WALLTIME=04:00:00
        ;;
    section2)
        NCPUS=24
        MEM=96GB
        WALLTIME=08:00:00
        ;;
    large)
        NCPUS=48
        MEM=190GB
        WALLTIME=24:00:00
        ;;
    *)
        echo "ERROR: Invalid PROFILE: $PROFILE"
        echo "Valid profiles: test, section1, section2, large"
        exit 1
        ;;
esac

# =============================================================================
# Validate Section Exists
# =============================================================================

# Convert section1_1 to section1/section1_1.py
# Extract section number: section1_1 -> section1
SECTION_DIR=$(echo "$SECTION" | sed 's/_[0-9].*//')
SECTION_SCRIPT="${SCRIPT_DIR}/${SECTION_DIR}/${SECTION}.py"

if [[ ! -f "$SECTION_SCRIPT" ]]; then
    echo "ERROR: Section script not found: $SECTION_SCRIPT"
    echo ""
    echo "Available sections:"
    find "${SCRIPT_DIR}/section1" "${SCRIPT_DIR}/section2" -name "section*.py" -type f 2>/dev/null | \
        sed 's|.*/||' | sed 's|\.py$||' | sort | sed 's/^/  /'
    exit 1
fi

# =============================================================================
# Display Submission Summary
# =============================================================================

echo ""
echo "=========================================="
echo "Submitting Experiment to PBS"
echo "=========================================="
echo "Section:   $SECTION"
echo "Epochs:    $EPOCHS"
echo "Profile:   $PROFILE"
echo "Resources: $NCPUS CPUs, $MEM memory"
echo "Walltime:  $WALLTIME"
echo "Script:    run_experiment.qsub"
echo "=========================================="
echo ""

# =============================================================================
# Submit to PBS with explicit resource flags
# =============================================================================

# Build qsub command with all necessary flags
QSUB_CMD="qsub"
QSUB_CMD="$QSUB_CMD -v SECTION=${SECTION},EPOCHS=${EPOCHS},PROFILE=${PROFILE}"
QSUB_CMD="$QSUB_CMD -l select=1:ncpus=${NCPUS}:mem=${MEM}"
QSUB_CMD="$QSUB_CMD -l walltime=${WALLTIME}"
QSUB_CMD="$QSUB_CMD ${SCRIPT_DIR}/run_experiment.qsub"

if [[ -n "$DRY_RUN" ]]; then
    echo "DRY RUN MODE - Would execute:"
    echo "$QSUB_CMD"
    echo ""
    echo "To actually submit, run without DRY_RUN=1"
else
    echo "Executing:"
    echo "$QSUB_CMD"
    echo ""

    # Submit the job
    JOB_ID=$(eval "$QSUB_CMD")
    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "✓ Job submitted successfully!"
        echo "  Job ID: $JOB_ID"
        echo ""
        echo "Monitor with:"
        echo "  qstat -u \$USER"
        echo "  qstat -f $JOB_ID"
        echo ""
        echo "View output when complete:"
        echo "  tail -f ~/pykan/madoc/pykan_experiment.o\${JOB_ID%%.*}"
    else
        echo "✗ Job submission failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi
fi

echo ""
echo "Submission complete."
