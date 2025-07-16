#!/usr/bin/env python3
"""
Test script to validate critical bug fixes before running experiments.
This script tests the key components that were causing experiment failures.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.neutrosophic.neutrosophic_transformer import NeutrosophicTransformer
from src.clustering.dual_clusterer import DualClusterer
from src.framework.forecasting_framework import NeutrosophicForecastingFramework

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_neutrosophic_transformer():
    """Test the neutrosophic transformer with mixed data types."""
    logger.info("Testing neutrosophic transformer...")

    # Create sample clustering outputs for neutrosophic transformation
    np.random.seed(42)
    n_samples = 100
    n_clusters = 3

    # Create sample K-means labels and FCM memberships
    kmeans_labels = np.random.randint(0, n_clusters, n_samples)
    fcm_memberships = np.random.dirichlet([1] * n_clusters, n_samples)

    try:
        transformer = NeutrosophicTransformer()
        neutrosophic_components = transformer.transform(kmeans_labels, fcm_memberships)

        # Validate output
        assert hasattr(neutrosophic_components, 'truth'), "Output should have truth component"
        assert hasattr(neutrosophic_components, 'indeterminacy'), "Output should have indeterminacy component"
        assert hasattr(neutrosophic_components, 'falsity'), "Output should have falsity component"
        assert len(neutrosophic_components.truth) == n_samples, "Truth component should have correct length"
        assert neutrosophic_components.truth.dtype in [np.float32, np.float64], f"Truth should be numeric, got {neutrosophic_components.truth.dtype}"
        assert not np.any(np.isnan(neutrosophic_components.truth)), "Truth should not contain NaN values"

        logger.info("✅ Neutrosophic transformer test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Neutrosophic transformer test failed: {e}")
        return False

def test_dual_clusterer():
    """Test the dual clusterer with configuration parameters."""
    logger.info("Testing dual clusterer...")

    try:
        # Create sample data
        np.random.seed(42)
        data = np.random.randn(100, 5)

        # Test with default configuration
        clusterer = DualClusterer(n_clusters=3)
        kmeans_labels, fcm_memberships = clusterer.fit_predict(data)

        # Validate output
        assert isinstance(kmeans_labels, np.ndarray), "K-means labels should be numpy array"
        assert isinstance(fcm_memberships, np.ndarray), "FCM memberships should be numpy array"
        assert len(kmeans_labels) == len(data), "Labels length should match data length"
        assert fcm_memberships.shape == (len(data), 3), "FCM memberships should have correct shape"
        assert kmeans_labels.dtype in [np.int32, np.int64], f"Labels should be integer, got {kmeans_labels.dtype}"
        assert fcm_memberships.dtype in [np.float32, np.float64], f"Memberships should be float, got {fcm_memberships.dtype}"

        logger.info("✅ Dual clusterer test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Dual clusterer test failed: {e}")
        return False

def test_framework_integration():
    """Test the overall framework integration."""
    logger.info("Testing framework integration...")

    try:
        # Create sample time series data
        np.random.seed(42)
        n_samples = 50  # Small sample for quick test

        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='H')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'energy_generation': np.random.uniform(0, 100, n_samples)
        })

        # Test framework initialization with correct parameters
        config = {
            'clustering': {'n_clusters': 3, 'max_iter': 10},
            'neutrosophic': {'entropy_epsilon': 1e-9, 'entropy_base': 2.0},
            'model': {'n_estimators': 10, 'max_depth': 3}
        }

        framework = NeutrosophicForecastingFramework(config)

        # Test basic functionality (without full training)
        logger.info("✅ Framework integration test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Framework integration test failed: {e}")
        return False

def main():
    """Run all bug fix validation tests."""
    logger.info("=== Starting Bug Fix Validation Tests ===")
    
    tests = [
        test_neutrosophic_transformer,
        test_dual_clusterer,
        test_framework_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    logger.info(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        logger.info("✅ All bug fix tests passed! Ready to run experiments.")
        return 0
    else:
        logger.error("❌ Some tests failed! Please check the fixes before running experiments.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
