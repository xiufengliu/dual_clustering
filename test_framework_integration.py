#!/usr/bin/env python3
"""
Test script to validate the framework integration with real datasets.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.framework.forecasting_framework import NeutrosophicForecastingFramework
from src.utils.math_utils import set_random_seeds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_framework_with_dataset(dataset_name):
    """Test framework with a specific dataset."""
    logger.info(f"=== Testing Framework with {dataset_name} ===")
    
    data_file = Path(f"data/processed/{dataset_name}.csv")
    if not data_file.exists():
        logger.warning(f"Dataset {dataset_name} not found, skipping")
        return False
    
    try:
        # Load and prepare data
        data = pd.read_csv(data_file)
        data['energy_generation'] = pd.to_numeric(data['energy_generation'], errors='coerce')
        data = data.dropna(subset=['energy_generation'])
        
        if len(data) < 100:
            logger.warning(f"Dataset {dataset_name} too small ({len(data)} rows), skipping")
            return False
        
        # Use a reasonable subset for testing
        test_data = data.head(min(1000, len(data))).copy()
        
        logger.info(f"Dataset {dataset_name}: shape={test_data.shape}, dtype={test_data['energy_generation'].dtype}")
        
        # Initialize framework
        set_random_seeds(42)
        framework = NeutrosophicForecastingFramework()
        
        # Test fitting
        framework.fit(test_data)
        logger.info(f"✅ Framework fitted successfully for {dataset_name}")
        
        # Test prediction
        predictions = framework.predict(horizon=5)
        logger.info(f"✅ Predictions generated for {dataset_name}: shape={predictions['predictions'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test framework with multiple datasets."""
    logger.info("Starting framework integration tests with real datasets...")
    
    # List of datasets to test
    datasets = [
        'entso_e_solar',
        'gefcom2014_energy', 
        'kaggle_solar_plant',
        'kaggle_wind_power',
        'nrel_solar',
        'uk_sheffield_solar'
    ]
    
    success_count = 0
    total_count = 0
    
    for dataset in datasets:
        total_count += 1
        if test_framework_with_dataset(dataset):
            success_count += 1
    
    logger.info(f"Integration test results: {success_count}/{total_count} datasets passed")
    
    if success_count > 0:
        logger.info("🎉 Framework integration tests passed! Ready for comprehensive evaluation.")
        return True
    else:
        logger.error("❌ All framework integration tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
