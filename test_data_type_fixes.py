#!/usr/bin/env python3
"""
Comprehensive test script to validate data type fixes for the neutrosophic transformation.
Tests with real datasets to ensure the bug fixes work correctly.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import modules directly
from src.neutrosophic.neutrosophic_transformer import NeutrosophicTransformer
from src.clustering.dual_clusterer import DualClusterer
from src.utils.math_utils import set_random_seeds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_type_conversion():
    """Test the _safe_convert_to_float64 method with various problematic data types."""
    logger.info("=== Testing Data Type Conversion ===")
    
    transformer = NeutrosophicTransformer()
    
    # Test 1: Mixed numeric and string data
    mixed_data = np.array([1.0, '2.5', 3, 'nan', '4.7', 'invalid'])
    result = transformer._safe_convert_to_float64(mixed_data, "mixed_test")
    logger.info(f"Mixed data conversion: {result}")
    assert result.dtype == np.float64
    assert len(result) == len(mixed_data)
    
    # Test 2: Unicode strings
    unicode_data = np.array(['1.5', '2.7', '3.9'], dtype='<U4')
    result = transformer._safe_convert_to_float64(unicode_data, "unicode_test")
    logger.info(f"Unicode data conversion: {result}")
    assert result.dtype == np.float64
    assert np.allclose(result, [1.5, 2.7, 3.9])
    
    # Test 3: Object array with mixed types
    object_data = np.array([1.0, '2.5', None, np.nan, 'text'], dtype=object)
    result = transformer._safe_convert_to_float64(object_data, "object_test")
    logger.info(f"Object data conversion: {result}")
    assert result.dtype == np.float64
    
    # Test 4: Empty array
    empty_data = np.array([])
    result = transformer._safe_convert_to_float64(empty_data, "empty_test")
    logger.info(f"Empty data conversion: {result}")
    assert result.dtype == np.float64
    
    logger.info("✅ Data type conversion tests passed!")

def test_dual_clusterer_robustness():
    """Test dual clusterer with various data types."""
    logger.info("=== Testing Dual Clusterer Robustness ===")
    
    set_random_seeds(42)
    
    # Create test data
    X = np.random.randn(100, 2).astype(np.float64)
    
    clusterer = DualClusterer(n_clusters=3, random_state=42)
    clusterer.fit(X)
    
    # Test integrated features
    integrated_features = clusterer.get_integrated_features()
    logger.info(f"Integrated features: shape={integrated_features.shape}, dtype={integrated_features.dtype}")
    
    assert integrated_features.dtype == np.float64
    assert integrated_features.shape == (100, 6)  # 3 clusters * 2 (one-hot + FCM)
    assert np.all(np.isfinite(integrated_features))
    
    # Test cluster assignments
    kmeans_labels, fcm_memberships = clusterer.get_cluster_assignments()
    logger.info(f"K-means labels: dtype={kmeans_labels.dtype}, FCM: dtype={fcm_memberships.dtype}")
    
    assert kmeans_labels.dtype.kind in ['i', 'u']  # Integer type
    assert fcm_memberships.dtype == np.float64
    assert np.allclose(np.sum(fcm_memberships, axis=1), 1.0, atol=1e-6)
    
    logger.info("✅ Dual clusterer robustness tests passed!")

def test_neutrosophic_transformation():
    """Test neutrosophic transformation with various inputs."""
    logger.info("=== Testing Neutrosophic Transformation ===")
    
    set_random_seeds(42)
    transformer = NeutrosophicTransformer()
    
    # Create test data
    n_samples, n_clusters = 50, 3
    kmeans_labels = np.random.randint(0, n_clusters, n_samples)
    
    # Create valid FCM memberships
    fcm_memberships = np.random.dirichlet([1] * n_clusters, n_samples)
    
    # Test transformation
    components = transformer.transform(kmeans_labels, fcm_memberships)
    
    logger.info(f"Neutrosophic components shapes: T={components.truth.shape}, I={components.indeterminacy.shape}, F={components.falsity.shape}")
    
    assert components.truth.dtype == np.float64
    assert components.indeterminacy.dtype == np.float64
    assert components.falsity.dtype == np.float64
    
    # Test enriched features creation
    original_features = np.random.randn(n_samples, 2).astype(np.float64)
    integrated_features = np.random.randn(n_samples, 6).astype(np.float64)
    
    enriched = transformer.create_enriched_features(
        original_features, integrated_features, components
    )
    
    logger.info(f"Enriched features: shape={enriched.shape}, dtype={enriched.dtype}")
    
    assert enriched.dtype == np.float64
    assert enriched.shape == (n_samples, 11)  # 2 + 6 + 3
    assert np.all(np.isfinite(enriched))
    
    logger.info("✅ Neutrosophic transformation tests passed!")

def test_with_real_dataset():
    """Test with a real dataset to ensure end-to-end functionality."""
    logger.info("=== Testing with Real Dataset ===")

    # Load a small real dataset
    data_file = Path("data/processed/entso_e_solar.csv")
    if not data_file.exists():
        logger.warning("Real dataset not found, skipping real dataset test")
        return

    try:
        # Load and prepare data
        data = pd.read_csv(data_file)
        data['energy_generation'] = pd.to_numeric(data['energy_generation'], errors='coerce')
        data = data.dropna(subset=['energy_generation'])

        # Use a small subset for testing
        test_data = data.head(200).copy()

        logger.info(f"Test data shape: {test_data.shape}")
        logger.info(f"Energy generation dtype: {test_data['energy_generation'].dtype}")

        # Test just the core components without full framework
        set_random_seeds(42)

        # Extract time series data
        values = test_data['energy_generation'].values.astype(np.float64)
        X = values.reshape(-1, 1)

        # Test dual clustering
        clusterer = DualClusterer(n_clusters=3, random_state=42)
        clusterer.fit(X)

        # Test integrated features
        integrated_features = clusterer.get_integrated_features()
        logger.info(f"Integrated features: shape={integrated_features.shape}, dtype={integrated_features.dtype}")

        # Test neutrosophic transformation
        kmeans_labels, fcm_memberships = clusterer.get_cluster_assignments()
        transformer = NeutrosophicTransformer()
        components = transformer.transform(kmeans_labels, fcm_memberships)

        # Test enriched features
        enriched = transformer.create_enriched_features(X, integrated_features, components)
        logger.info(f"Enriched features: shape={enriched.shape}, dtype={enriched.dtype}")

        logger.info("✅ Real dataset test passed!")

    except Exception as e:
        logger.error(f"Real dataset test failed: {e}")
        raise

def main():
    """Run all tests."""
    logger.info("Starting comprehensive data type fix validation tests...")
    
    try:
        test_data_type_conversion()
        test_dual_clusterer_robustness()
        test_neutrosophic_transformation()
        test_with_real_dataset()
        
        logger.info("🎉 All tests passed! Data type fixes are working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
