"""
Usage Examples for Refactored Analysis Module

This script demonstrates the simplified usage of the analysis tools
after the data_io refactoring.

Run from section1 directory:
    cd /path/to/section1
    python analysis/USAGE_EXAMPLE.py
"""

import sys
from pathlib import Path

# Add section1 to path so we can import analysis
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import data_io
from analysis import MetricsAnalyzer, FunctionFittingVisualizer, Heatmap2DAnalyzer


def example_1_check_available_results():
    """Example 1: Check what results are available"""
    print("=" * 70)
    print("Example 1: Check Available Results")
    print("=" * 70)

    data_io.print_available_results()


def example_2_load_results_directly():
    """Example 2: Load results directly using data_io"""
    print("\n" + "=" * 70)
    print("Example 2: Load Results Using data_io")
    print("=" * 70)

    # Load latest results for section1_1
    results, metadata = data_io.load_results('section1_1')

    print(f"\nLoaded results for section1_1:")
    print(f"  Model types: {list(results.keys())}")
    print(f"  Number of datasets: {len(results['mlp'])}")
    print(f"  Metadata available: {'Yes' if metadata else 'No'}")

    # Get complete section data including models paths
    section_data = data_io.load_section_results('section1_1')
    print(f"\nComplete section data:")
    print(f"  Timestamp: {section_data['timestamp']}")
    print(f"  Results file: {section_data['results_file'].name}")
    print(f"  Models directory: {section_data['models_dir'].name if section_data['models_dir'] else 'Not found'}")


def example_3_simple_analysis():
    """Example 3: Run analysis with minimal code"""
    print("\n" + "=" * 70)
    print("Example 3: Simple Analysis with Section ID")
    print("=" * 70)

    # Before refactoring, you needed:
    # analyzer = MetricsAnalyzer('/path/to/sec1_results/section1_1_results_20251021_215324.pkl')

    # Now you can just use:
    analyzer = MetricsAnalyzer('section1_1')  # Auto-loads latest results!

    print(f"\n✓ Created MetricsAnalyzer")
    print(f"  Loaded: {analyzer.results_path.name}")
    print(f"  Metadata: {'Available' if analyzer.metadata else 'None'}")

    # Generate a comparison table
    df = analyzer.create_comparison_table(dataset_idx=0)
    print(f"\n✓ Created comparison table:")
    print(df.head())


def example_4_with_auto_discovery():
    """Example 4: Auto-discovery of model directories"""
    print("\n" + "=" * 70)
    print("Example 4: Auto-Discovery of Models")
    print("=" * 70)

    # Before refactoring:
    # visualizer = FunctionFittingVisualizer(
    #     '/path/to/results.pkl',
    #     models_dir='/path/to/kan_models_TIMESTAMP'
    # )

    # Now: models directory is auto-discovered!
    visualizer = FunctionFittingVisualizer('section1_1')

    print(f"\n✓ Created FunctionFittingVisualizer")
    print(f"  Results: {visualizer.results_path.name}")
    print(f"  Models: {visualizer.models_dir.name if visualizer.models_dir else 'Not found'}")
    print(f"  Metadata: {'Available' if visualizer.metadata else 'None'}")


def example_5_explicit_paths_still_work():
    """Example 5: Explicit paths still work for backward compatibility"""
    print("\n" + "=" * 70)
    print("Example 5: Explicit Paths (Backward Compatible)")
    print("=" * 70)

    # Find the actual path
    info = data_io.find_latest_results('section1_1')

    # You can still use explicit paths if needed
    analyzer = MetricsAnalyzer(str(info['results_file']))

    print(f"\n✓ Created analyzer with explicit path")
    print(f"  Path: {analyzer.results_path}")


def example_6_working_with_2d_data():
    """Example 6: Working with 2D data (section1_3)"""
    print("\n" + "=" * 70)
    print("Example 6: 2D Data Analysis")
    print("=" * 70)

    try:
        # Load 2D results
        heatmap_analyzer = Heatmap2DAnalyzer('section1_3')

        print(f"\n✓ Created Heatmap2DAnalyzer for 2D data")
        print(f"  Results: {heatmap_analyzer.results_path.name}")
        print(f"  Models: {heatmap_analyzer.models_dir.name if heatmap_analyzer.models_dir else 'Not found'}")
        print(f"  Functions: {list(heatmap_analyzer.function_map.keys())}")
    except Exception as e:
        print(f"\n✗ Error loading 2D data: {e}")


def example_7_find_specific_timestamp():
    """Example 7: Load specific timestamp instead of latest"""
    print("\n" + "=" * 70)
    print("Example 7: Load Specific Timestamp")
    print("=" * 70)

    # Find all available timestamps first
    import pathlib
    results_dir = data_io.get_section_results_dir('section1_1')
    all_results = list(results_dir.glob('section1_1_results_*.pkl'))

    if len(all_results) > 0:
        # Extract timestamps
        timestamps = [f.stem.split('_')[-2:] for f in all_results]
        timestamps = ['_'.join(ts) for ts in timestamps]

        print(f"\nAvailable timestamps: {timestamps}")

        # Load specific timestamp
        results, metadata = data_io.load_results(
            'section1_1',
            timestamp=timestamps[0]
        )
        print(f"\n✓ Loaded specific timestamp: {timestamps[0]}")
        print(f"  Model types: {list(results.keys())}")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("ANALYSIS MODULE USAGE EXAMPLES")
    print("After data_io Refactoring")
    print("=" * 70)

    examples = [
        example_1_check_available_results,
        example_2_load_results_directly,
        example_3_simple_analysis,
        example_4_with_auto_discovery,
        example_5_explicit_paths_still_work,
        example_6_working_with_2d_data,
        example_7_find_specific_timestamp,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Error in {example_func.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nKey Benefits of Refactoring:")
    print("  1. ✓ Auto-discovery of latest results")
    print("  2. ✓ Auto-discovery of model directories")
    print("  3. ✓ Auto-discovery of metadata files")
    print("  4. ✓ Simple section IDs instead of long paths")
    print("  5. ✓ Centralized, tested IO logic")
    print("  6. ✓ Backward compatible with explicit paths")
    print()


if __name__ == '__main__':
    main()
