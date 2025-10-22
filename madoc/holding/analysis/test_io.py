"""
Test script for io.py module

This script tests the IO module with actual data from the repository
to ensure all functions work correctly before refactoring the analysis files.
"""

import sys
from pathlib import Path

# Import our data_io module (renamed from io.py to avoid conflict with built-in)
import data_io


def test_section_results_dir():
    """Test getting section results directories"""
    print("\n" + "="*70)
    print("TEST: get_section_results_dir()")
    print("="*70)

    for section_id in ['section1_1', 'section1_2', 'section1_3']:
        try:
            results_dir = data_io.get_section_results_dir(section_id)
            exists = results_dir.exists()
            print(f"✓ {section_id}: {results_dir} (exists: {exists})")
        except Exception as e:
            print(f"✗ {section_id}: {e}")

    # Test invalid section
    try:
        data_io.get_section_results_dir('invalid_section')
        print("✗ Should have raised InvalidSectionError")
    except data_io.InvalidSectionError as e:
        print(f"✓ Invalid section correctly rejected: {e}")


def test_find_latest_results():
    """Test finding latest results for each section"""
    print("\n" + "="*70)
    print("TEST: find_latest_results()")
    print("="*70)

    for section_id in ['section1_1', 'section1_2', 'section1_3']:
        print(f"\n{section_id}:")
        try:
            info = data_io.find_latest_results(section_id)
            print(f"  ✓ Found results")
            print(f"    Timestamp: {info['timestamp']}")
            print(f"    Results: {info['results_file'].name}")
            print(f"    Metadata: {info['metadata_file'].name if info['metadata_file'] else 'Not found'}")
            print(f"    Models: {info['models_dir'].name if info['models_dir'] else 'Not found'}")
            print(f"    Pruned: {info['pruned_models_dir'].name if info['pruned_models_dir'] else 'Not found'}")
        except data_io.ResultsNotFoundError as e:
            print(f"  ✗ {e}")


def test_load_results_file():
    """Test loading results files"""
    print("\n" + "="*70)
    print("TEST: load_results_file()")
    print("="*70)

    for section_id in ['section1_1', 'section1_2', 'section1_3']:
        try:
            info = data_io.find_latest_results(section_id)
            results = data_io.load_results_file(info['results_file'])

            print(f"\n{section_id}:")
            print(f"  ✓ Loaded {info['results_file'].name}")
            print(f"    Type: {type(results)}")
            print(f"    Keys: {list(results.keys())}")

            # Check data structure
            for model_type in ['mlp', 'siren', 'kan', 'kan_pruning']:
                if model_type in results:
                    num_datasets = len(results[model_type])
                    print(f"    {model_type}: {num_datasets} datasets")
        except data_io.ResultsNotFoundError:
            print(f"\n{section_id}: ✗ No results found")
        except Exception as e:
            print(f"\n{section_id}: ✗ Error: {e}")


def test_load_metadata():
    """Test loading metadata files"""
    print("\n" + "="*70)
    print("TEST: load_metadata_file()")
    print("="*70)

    for section_id in ['section1_1', 'section1_2', 'section1_3']:
        try:
            info = data_io.find_latest_results(section_id)
            if info['metadata_file']:
                metadata = data_io.load_metadata_file(info['metadata_file'])
                print(f"\n{section_id}:")
                print(f"  ✓ Loaded {info['metadata_file'].name}")
                print(f"    Keys: {list(metadata.keys())}")
            else:
                print(f"\n{section_id}: No metadata file found")
        except data_io.ResultsNotFoundError:
            print(f"\n{section_id}: ✗ No results found")


def test_load_results_convenience():
    """Test the main convenience function load_results()"""
    print("\n" + "="*70)
    print("TEST: load_results() with section IDs")
    print("="*70)

    for section_id in ['section1_1', 'section1_2', 'section1_3']:
        try:
            results, metadata = data_io.load_results(section_id)
            print(f"\n{section_id}:")
            print(f"  ✓ Loaded results and metadata")
            print(f"    Results type: {type(results)}")
            print(f"    Results keys: {list(results.keys())}")
            print(f"    Metadata: {'Available' if metadata else 'None'}")
        except data_io.ResultsNotFoundError as e:
            print(f"\n{section_id}: ✗ {e}")


def test_load_results_with_path():
    """Test load_results() with explicit paths"""
    print("\n" + "="*70)
    print("TEST: load_results() with explicit paths")
    print("="*70)

    for section_id in ['section1_1', 'section1_2', 'section1_3']:
        try:
            # First find the path
            info = data_io.find_latest_results(section_id)

            # Now load using explicit path
            results, metadata = data_io.load_results(str(info['results_file']))
            print(f"\n{section_id}:")
            print(f"  ✓ Loaded from explicit path: {info['results_file'].name}")
            print(f"    Metadata: {'Available' if metadata else 'None'}")
        except data_io.ResultsNotFoundError:
            print(f"\n{section_id}: ✗ No results found")


def test_load_section_results():
    """Test the comprehensive load_section_results() function"""
    print("\n" + "="*70)
    print("TEST: load_section_results()")
    print("="*70)

    for section_id in ['section1_1', 'section1_2', 'section1_3']:
        try:
            data = data_io.load_section_results(section_id)
            print(f"\n{section_id}:")
            print(f"  ✓ Complete data loaded")
            print(f"    Timestamp: {data['timestamp']}")
            print(f"    Results: {type(data['results'])}")
            print(f"    Metadata: {'Available' if data['metadata'] else 'None'}")
            print(f"    Models dir: {'Available' if data['models_dir'] else 'None'}")
            print(f"    Pruned models dir: {'Available' if data['pruned_models_dir'] else 'None'}")
        except data_io.ResultsNotFoundError as e:
            print(f"\n{section_id}: ✗ {e}")


def test_find_models_dir():
    """Test finding models directories"""
    print("\n" + "="*70)
    print("TEST: find_models_dir()")
    print("="*70)

    for section_id in ['section1_1', 'section1_2', 'section1_3']:
        try:
            info = data_io.find_latest_results(section_id)

            # Test regular models
            models_dir = data_io.find_models_dir(info['results_file'])
            pruned_dir = data_io.find_models_dir(info['results_file'], pruned=True)

            print(f"\n{section_id}:")
            print(f"  Models dir: {'✓ ' + str(models_dir.name) if models_dir else '✗ Not found'}")
            print(f"  Pruned dir: {'✓ ' + str(pruned_dir.name) if pruned_dir else '✗ Not found'}")
        except data_io.ResultsNotFoundError:
            print(f"\n{section_id}: ✗ No results found")


def test_print_available_results():
    """Test the summary print function"""
    print("\n" + "="*70)
    print("TEST: print_available_results()")
    print("="*70)

    data_io.print_available_results()


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("ANALYSIS IO MODULE TEST SUITE")
    print("="*70)

    tests = [
        test_section_results_dir,
        test_find_latest_results,
        test_load_results_file,
        test_load_metadata,
        test_load_results_convenience,
        test_load_results_with_path,
        test_load_section_results,
        test_find_models_dir,
        test_print_available_results,
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_func.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)


if __name__ == '__main__':
    run_all_tests()
