#!/bin/bash

# plot.sh - Generate all visualizations and tables
# This script runs all plotting and table generation scripts for both sections
#
# Usage:
#   ./plot.sh                    # Generate all plots and tables
#   ./plot.sh --section1-only    # Generate only Section 1 plots and tables
#   ./plot.sh --section2-only    # Generate only Section 2 plots and tables
#   ./plot.sh --plots-only       # Generate only plots (skip tables)
#   ./plot.sh --tables-only      # Generate only tables (skip plots)
#   ./plot.sh --continue-on-error # Continue even if some scripts fail

# Parse command line arguments
SECTION1=true
SECTION2=true
PLOTS=true
TABLES=true
CONTINUE_ON_ERROR=false
RUN_LABEL=""

for arg in "$@"; do
    case $arg in
        --section1-only)
            SECTION2=false
            ;;
        --section2-only)
            SECTION1=false
            ;;
        --plots-only)
            TABLES=false
            ;;
        --tables-only)
            PLOTS=false
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            ;;
        --run-label=*)
            RUN_LABEL="${arg#*=}"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --section1-only       Generate only Section 1 plots and tables"
            echo "  --section2-only       Generate only Section 2 plots and tables"
            echo "  --plots-only          Generate only plots (skip tables)"
            echo "  --tables-only         Generate only tables (skip plots)"
            echo "  --continue-on-error   Continue even if some scripts fail"
            echo "  --run-label=LABEL     Custom label for this run (default: 'results')"
            echo "  --help, -h            Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set error handling based on flags
if [ "$CONTINUE_ON_ERROR" = false ]; then
    set -e  # Exit on any error
fi

# Create run timestamp and label
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -z "$RUN_LABEL" ]; then
    RUN_LABEL="results"
fi
RUN_FOLDER="${RUN_TIMESTAMP}_${RUN_LABEL}"

echo "=========================================="
echo "Starting Visualization and Table Generation"
echo "=========================================="
echo "Configuration:"
echo "  Section 1: $SECTION1"
echo "  Section 2: $SECTION2"
echo "  Plots: $PLOTS"
echo "  Tables: $TABLES"
echo "  Continue on error: $CONTINUE_ON_ERROR"
echo "  Run folder: $RUN_FOLDER"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create output directories for this run
if [ "$SECTION1" = true ] && [ "$PLOTS" = true ]; then
    SECTION1_VIZ_DIR="$SCRIPT_DIR/section1/visualization/outputs/$RUN_FOLDER"
    mkdir -p "$SECTION1_VIZ_DIR"
    echo "Created Section 1 visualization output directory: $SECTION1_VIZ_DIR"
fi

if [ "$SECTION1" = true ] && [ "$TABLES" = true ]; then
    SECTION1_TABLE_DIR="$SCRIPT_DIR/section1/tables/outputs/$RUN_FOLDER"
    mkdir -p "$SECTION1_TABLE_DIR"
    echo "Created Section 1 tables output directory: $SECTION1_TABLE_DIR"
fi

if [ "$SECTION2" = true ] && [ "$PLOTS" = true ]; then
    SECTION2_VIZ_DIR="$SCRIPT_DIR/section2/visualization/outputs/$RUN_FOLDER"
    mkdir -p "$SECTION2_VIZ_DIR"
    echo "Created Section 2 visualization output directory: $SECTION2_VIZ_DIR"
fi
echo ""

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

# Track successes and failures
SUCCESSES=0
FAILURES=0
FAILED_SCRIPTS=()

# Helper function to run a command and track success/failure
run_script() {
    local description="$1"
    local command="$2"

    echo "Running: $description"
    if eval "$command"; then
        echo "✓ $description"
        ((SUCCESSES++))
    else
        echo "✗ FAILED: $description"
        ((FAILURES++))
        FAILED_SCRIPTS+=("$description")
        if [ "$CONTINUE_ON_ERROR" = false ]; then
            exit 1
        fi
    fi
    echo ""
}

# ============================================================================
# SECTION 1 - VISUALIZATIONS
# ============================================================================

if [ "$SECTION1" = true ] && [ "$PLOTS" = true ]; then
    echo "=========================================="
    echo "SECTION 1 - VISUALIZATIONS"
    echo "=========================================="
    echo ""

    # Section 1.1 - Function Approximation
    echo "=== Section 1.1: Function Approximation Plots ==="
    run_script "Section 1.1 - Best loss curves" \
        "$PYTHON '$SCRIPT_DIR/section1/visualization/plot_best_loss.py' --section section1_1 --loss-type test --output-dir '$SECTION1_VIZ_DIR'"

    run_script "Section 1.1 - Function fit plots" \
        "$PYTHON '$SCRIPT_DIR/section1/visualization/plot_function_fit.py' --section section1_1 --output-dir '$SECTION1_VIZ_DIR'"

    # Note: Checkpoint comparison script is not compatible with current checkpoint format
    # run_script "Section 1.1 - Checkpoint comparison" \
    #     "$PYTHON '$SCRIPT_DIR/section1/visualization/plot_checkpoint_comparison.py' --section section1_1 --output-dir '$SECTION1_VIZ_DIR'"

    # Section 1.2 - 1D Poisson PDE
    echo "=== Section 1.2: 1D Poisson PDE Plots ==="
    run_script "Section 1.2 - Best loss curves" \
        "$PYTHON '$SCRIPT_DIR/section1/visualization/plot_best_loss.py' --section section1_2 --loss-type test --output-dir '$SECTION1_VIZ_DIR'"

    run_script "Section 1.2 - Function fit plots" \
        "$PYTHON '$SCRIPT_DIR/section1/visualization/plot_function_fit.py' --section section1_2 --output-dir '$SECTION1_VIZ_DIR'"

    # Section 1.3 - 2D Poisson PDE
    echo "=== Section 1.3: 2D Poisson PDE Plots ==="
    run_script "Section 1.3 - Best loss curves" \
        "$PYTHON '$SCRIPT_DIR/section1/visualization/plot_best_loss.py' --section section1_3 --loss-type test --output-dir '$SECTION1_VIZ_DIR'"

    run_script "Section 1.3 - Function fit plots" \
        "$PYTHON '$SCRIPT_DIR/section1/visualization/plot_function_fit.py' --section section1_3 --output-dir '$SECTION1_VIZ_DIR'"

    run_script "Section 1.3 - 2D heatmap visualizations" \
        "$PYTHON '$SCRIPT_DIR/section1/visualization/plot_heatmap_2d.py' --output-dir '$SECTION1_VIZ_DIR'"
fi

# ============================================================================
# SECTION 1 - TABLES
# ============================================================================

if [ "$SECTION1" = true ] && [ "$TABLES" = true ]; then
    echo "=========================================="
    echo "SECTION 1 - TABLES"
    echo "=========================================="
    echo ""

    run_script "Section 1 - All tables" \
        "$PYTHON '$SCRIPT_DIR/section1/tables/generate_all_tables.py' --output-dir '$SECTION1_TABLE_DIR'"
fi

# ============================================================================
# SECTION 2 - VISUALIZATIONS
# ============================================================================

if [ "$SECTION2" = true ] && [ "$PLOTS" = true ]; then
    echo "=========================================="
    echo "SECTION 2 - VISUALIZATIONS"
    echo "=========================================="
    echo ""

    # Section 2.1 - Optimizer Comparison
    echo "=== Section 2.1: Optimizer Comparison Plots ==="
    run_script "Section 2.1 - Best loss curves" \
        "$PYTHON '$SCRIPT_DIR/section2/visualization/plot_best_loss.py' --section section2_1 --metric dense_mse"

    run_script "Section 2.1 - Optimizer comparison plots" \
        "$PYTHON '$SCRIPT_DIR/section2/visualization/plot_optimizer_comparison.py' --section section2_1 --plot-type both"

    run_script "Section 2.1 - Function fit plots" \
        "$PYTHON '$SCRIPT_DIR/section2/visualization/plot_function_fit.py' --section section2_1"

    run_script "Section 2.1 - 2D heatmap visualizations" \
        "$PYTHON '$SCRIPT_DIR/section2/visualization/plot_heatmap_2d.py' --section section2_1"

    run_script "Section 2.1 - 1D cross-section plots" \
        "$PYTHON '$SCRIPT_DIR/section2/visualization/plot_cross_sections_1d.py' --section section2_1"

    # Section 2.2 - Adaptive Grid
    echo "=== Section 2.2: Adaptive Grid Plots ==="
    run_script "Section 2.2 - Best loss curves" \
        "$PYTHON '$SCRIPT_DIR/section2/visualization/plot_best_loss.py' --section section2_2 --metric dense_mse"

    # High-D experiments (if results exist)
    echo "=== High-Dimensional Experiments ==="

    # Check if high-D results exist
    HIGHD_RESULTS_3D="$SCRIPT_DIR/section2/results/sec1_highd_results"
    if [ -d "$HIGHD_RESULTS_3D" ]; then
        run_script "High-D - Dimension comparison plots" \
            "$PYTHON '$SCRIPT_DIR/section2/visualization/plot_dimension_comparison.py' --section section2_1 --plot-type both"

        run_script "High-D - Scaling laws (deep)" \
            "$PYTHON '$SCRIPT_DIR/section2/visualization/plot_scaling_laws.py' --dim all --architecture deep"

        run_script "High-D - Scaling laws (shallow)" \
            "$PYTHON '$SCRIPT_DIR/section2/visualization/plot_scaling_laws.py' --dim all --architecture shallow"
    else
        echo "⚠ Skipping high-D plots (no results found at $HIGHD_RESULTS_3D)"
        echo "  Run section2_1_highd.py and section2_2_highd.py first to generate high-D plots"
        echo ""
    fi
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo "=========================================="
echo "Generation Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Successful: $SUCCESSES"
echo "  Failed: $FAILURES"
echo ""

if [ $FAILURES -gt 0 ]; then
    echo "Failed scripts:"
    for script in "${FAILED_SCRIPTS[@]}"; do
        echo "  ✗ $script"
    done
    echo ""
fi

echo "Generated artifacts:"
if [ "$SECTION1" = true ]; then
    if [ "$PLOTS" = true ]; then
        echo "  Section 1 Visualizations: section1/visualization/outputs/$RUN_FOLDER/"
    fi
    if [ "$TABLES" = true ]; then
        echo "  Section 1 Tables:         section1/tables/outputs/$RUN_FOLDER/"
    fi
fi
if [ "$SECTION2" = true ] && [ "$PLOTS" = true ]; then
    echo "  Section 2 Visualizations: section2/visualization/outputs/$RUN_FOLDER/"
fi
echo ""
echo "Note: All plots are saved to files by default (no popups)"
echo "To display plots in a window, add --show flag to individual scripts"
echo "Example: python section1/visualization/plot_best_loss.py --section section1_1 --show"
echo ""

# Exit with appropriate code
if [ $FAILURES -gt 0 ]; then
    exit 1
else
    exit 0
fi
