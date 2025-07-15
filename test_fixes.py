#!/usr/bin/env python3
"""
Test script to validate bug fixes and optimizations.
This script tests the core components with minimal data to ensure they work correctly.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.neutrosophic.neutrosophic_transformer import NeutrosophicTransformer, NeutrosophicComponents
from src.clustering.dual_clusterer import DualClusterer
from src.utils.logger import setup_logger

def test_neutrosophic_transformer():
    """Test the neutrosophic transformer with various data types."""
    print("Testing Neutrosophic Transformer...")
    
    # Setup logger
    logger = setup_logger("test_neutrosophic", level=logging.DEBUG)
    
    # Create test data with potential problematic types
    n_samples = 50
    n_clusters = 3
    
    # Test 1: Normal numeric data
    print("Test 1: Normal numeric data")
    kmeans_labels = np.random.randint(0, n_clusters, n_samples)
    fcm_memberships = np.random.dirichlet(np.ones(n_clusters), n_samples)
    
    transformer = NeutrosophicTransformer()
    try:
        components = transformer.transform(kmeans_labels, fcm_memberships)
        print(f"✓ Normal data test passed: {components.truth.shape}")
    except Exception as e:
        print(f"✗ Normal data test failed: {e}")
        return False
    
    # Test 2: Mixed data types (the problematic case)
    print("Test 2: Mixed data types")
    try:
        # Create arrays with mixed types (simulating the bug)
        mixed_kmeans = np.array([0, 1, 2, 'bad', 1, 0] * (n_samples // 6 + 1))[:n_samples]
        mixed_fcm = np.random.dirichlet(np.ones(n_clusters), n_samples)
        
        # This should handle the mixed types gracefully
        components = transformer.transform(mixed_kmeans, mixed_fcm)
        print(f"✓ Mixed data test passed: {components.truth.shape}")
    except Exception as e:
        print(f"✗ Mixed data test failed: {e}")
        return False
    
    # Test 3: Feature enrichment
    print("Test 3: Feature enrichment")
    try:
        original_features = np.random.randn(n_samples, 5)
        integrated_features = np.random.randn(n_samples, n_clusters * 2)
        
        enriched = transformer.create_enriched_features(
            original_features, integrated_features, components
        )
        expected_features = 5 + n_clusters * 2 + 3  # original + integrated + neutrosophic
        print(f"✓ Feature enrichment test passed: {enriched.shape} (expected {n_samples} x {expected_features})")
    except Exception as e:
        print(f"✗ Feature enrichment test failed: {e}")
        return False
    
    return True

def test_dual_clusterer():
    """Test the dual clusterer with optimized parameters."""
    print("Testing Dual Clusterer...")
    
    # Create test data
    n_samples = 100
    n_features = 3
    X = np.random.randn(n_samples, n_features)
    
    # Test with optimized parameters
    clusterer = DualClusterer(
        n_clusters=3,
        fcm_fuzziness=2.0,
        max_iter=20,  # Reduced for speed
        tol=0.1,      # Relaxed for speed
        random_state=42
    )
    
    try:
        clusterer.fit(X)
        kmeans_labels, fcm_memberships = clusterer.predict(X)
        print(f"✓ Dual clusterer test passed: labels={kmeans_labels.shape}, memberships={fcm_memberships.shape}")
        return True
    except Exception as e:
        print(f"✗ Dual clusterer test failed: {e}")
        return False

def test_data_type_conversion():
    """Test comprehensive data type conversion."""
    print("Testing Data Type Conversion...")
    
    transformer = NeutrosophicTransformer()
    
    # Test various problematic data types
    test_cases = [
        np.array([1, 2, 3, 4, 5]),  # Normal integers
        np.array([1.0, 2.0, 3.0]),  # Normal floats
        np.array(['1', '2', '3']),  # String numbers
        np.array([1, '2', 3.0, 'bad', 5]),  # Mixed types
        np.array([np.nan, np.inf, -np.inf, 1.0]),  # Special values
    ]
    
    for i, test_array in enumerate(test_cases):
        try:
            converted = transformer._comprehensive_float64_conversion(test_array, f"test_case_{i}")
            print(f"✓ Test case {i} passed: {test_array.dtype} -> {converted.dtype}")
        except Exception as e:
            print(f"✗ Test case {i} failed: {e}")
            return False
    
    return True

def test_small_dataset_processing():
    """Test processing with a small synthetic dataset."""
    print("Testing Small Dataset Processing...")
    
    # Create a small synthetic dataset
    n_samples = 100
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    energy_generation = np.random.exponential(scale=100, size=n_samples)
    
    # Add some realistic patterns
    hourly_pattern = 50 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    energy_generation += hourly_pattern
    energy_generation = np.maximum(0, energy_generation)  # Ensure non-negative
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'energy_generation': energy_generation
    })
    
    print(f"Created synthetic dataset: {len(data)} samples")
    print(f"Energy range: {energy_generation.min():.2f} - {energy_generation.max():.2f}")
    
    # Test basic processing steps
    try:
        # Step 1: Dual clustering
        X = energy_generation.reshape(-1, 1)
        clusterer = DualClusterer(n_clusters=3, max_iter=20, tol=0.1)
        clusterer.fit(X)
        kmeans_labels, fcm_memberships = clusterer.predict(X)
        
        # Step 2: Neutrosophic transformation
        transformer = NeutrosophicTransformer()
        components = transformer.transform(kmeans_labels, fcm_memberships)
        
        # Step 3: Feature enrichment
        original_features = X
        integrated_features = np.column_stack([
            np.eye(3)[kmeans_labels],  # One-hot encoded K-means
            fcm_memberships            # FCM memberships
        ])
        
        enriched_features = transformer.create_enriched_features(
            original_features, integrated_features, components
        )
        
        print(f"✓ Small dataset processing passed:")
        print(f"  - Original features: {original_features.shape}")
        print(f"  - Integrated features: {integrated_features.shape}")
        print(f"  - Neutrosophic components: {components.truth.shape}")
        print(f"  - Enriched features: {enriched_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Small dataset processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("TESTING BUG FIXES AND OPTIMIZATIONS")
    print("="*60)
    
    tests = [
        test_data_type_conversion,
        test_neutrosophic_transformer,
        test_dual_clusterer,
        test_small_dataset_processing,
    ]
    
    results = []
    for test_func in tests:
        print(f"\n{'-'*40}")
        result = test_func()
        results.append(result)
        print(f"Result: {'PASS' if result else 'FAIL'}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The fixes are working correctly.")
        print("\nNext steps:")
        print("1. Run optimized experiments with: python run_optimized_experiments.py --config debug_config")
        print("2. Use small datasets first to validate functionality")
        print("3. Gradually increase dataset sizes for performance testing")
    else:
        print("❌ Some tests failed. Check the error messages above.")
        print("Review the neutrosophic transformer implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)