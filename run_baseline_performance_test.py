#!/usr/bin/env python3
"""
Baseline performance test focusing on the core optimizations.
This script tests the performance improvements without complex neutrosophic components.
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.baseline_models import BaselineForecasters
from src.evaluation.metrics import ForecastingMetrics
from src.utils.logger import setup_logger

def create_features(data: pd.DataFrame, n_lags: int = 24) -> tuple:
    """Create lag features for baseline models."""
    X_list = []
    y_list = []
    
    for i in range(n_lags, len(data)):
        X_list.append(data['energy_generation'].iloc[i-n_lags:i].values)
        y_list.append(data['energy_generation'].iloc[i])
    
    return np.array(X_list), np.array(y_list)

def prepare_dataset(dataset_name: str, max_samples: int = 5000):
    """Prepare dataset with size limit."""
    data_file = Path("data/processed") / f"{dataset_name}.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")
    
    data = pd.read_csv(data_file)
    
    # Apply size limit
    if len(data) > max_samples:
        data = data.iloc[-max_samples:]
    
    # Ensure timestamp column
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')
    
    # Clean data
    data = data.dropna()
    data['energy_generation'] = pd.to_numeric(data['energy_generation'], errors='coerce')
    data = data.dropna()
    
    # Split data
    n_total = len(data)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    
    train_data = data.iloc[:n_train]
    val_data = data.iloc[n_train:n_train + n_val]
    test_data = data.iloc[n_train + n_val:]
    
    return train_data, val_data, test_data

def run_baseline_performance_test():
    """Run baseline performance test."""
    # Setup logging
    log_file = Path("results/optimized") / f"baseline_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("baseline_performance", log_file=log_file)
    
    logger.info("="*80)
    logger.info("BASELINE PERFORMANCE TEST")
    logger.info("="*80)
    
    # Test datasets
    datasets = ['kaggle_solar_plant', 'entso_e_load', 'nrel_canada_wind']
    max_samples = 2000  # Smaller for faster testing
    
    # Initialize metrics calculator
    metrics_calculator = ForecastingMetrics()
    
    # Results storage
    results = {}
    total_start_time = time.time()
    
    for dataset in datasets:
        logger.info(f"\n{'='*50}")
        logger.info(f"TESTING DATASET: {dataset}")
        logger.info(f"{'='*50}")
        
        dataset_start_time = time.time()
        
        try:
            # Prepare data
            train_data, val_data, test_data = prepare_dataset(dataset, max_samples)
            logger.info(f"Data prepared: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
            
            # Create features
            X_train, y_train = create_features(train_data)
            X_test, y_test = create_features(test_data)
            logger.info(f"Features created: X_train={X_train.shape}, X_test={X_test.shape}")
            
            # Get baseline models
            baseline_models = BaselineForecasters.create_all_models(random_state=42)
            
            dataset_results = {}
            
            for model_name, model in baseline_models.items():
                logger.info(f"\nTesting {model_name}...")
                model_start_time = time.time()
                
                try:
                    # Train model
                    train_start = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - train_start
                    
                    # Make predictions
                    pred_start = time.time()
                    predictions, uncertainties = model.predict_with_uncertainty(X_test)
                    prediction_time = time.time() - pred_start
                    
                    # Calculate metrics
                    point_metrics = metrics_calculator.calculate_point_metrics(y_test, predictions)
                    
                    model_total_time = time.time() - model_start_time
                    
                    dataset_results[model_name] = {
                        'point_metrics': point_metrics,
                        'training_time': training_time,
                        'prediction_time': prediction_time,
                        'total_time': model_total_time,
                        'success': True
                    }
                    
                    logger.info(f"  {model_name} - RMSE: {point_metrics['rmse']:.4f}, Time: {model_total_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"  {model_name} failed: {e}")
                    dataset_results[model_name] = {
                        'error': str(e),
                        'success': False
                    }
            
            dataset_time = time.time() - dataset_start_time
            results[dataset] = {
                'model_results': dataset_results,
                'dataset_time': dataset_time,
                'data_info': {
                    'train_samples': len(train_data),
                    'val_samples': len(val_data),
                    'test_samples': len(test_data)
                }
            }
            
            logger.info(f"\nDataset {dataset} completed in {dataset_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Dataset {dataset} failed: {e}")
            results[dataset] = {'error': str(e), 'failed': True}
    
    total_time = time.time() - total_start_time
    
    # Performance summary
    logger.info(f"\n{'='*80}")
    logger.info("PERFORMANCE SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"Datasets processed: {len(datasets)}")
    logger.info(f"Average time per dataset: {total_time/len(datasets):.2f} seconds")
    
    # Results summary
    logger.info(f"\n{'='*80}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    
    for dataset, dataset_results in results.items():
        if 'model_results' in dataset_results:
            logger.info(f"\n{dataset.upper()}:")
            for model_name, model_results in dataset_results['model_results'].items():
                if model_results.get('success', False):
                    rmse = model_results['point_metrics']['rmse']
                    time_taken = model_results['total_time']
                    logger.info(f"  {model_name}: RMSE = {rmse:.4f}, Time = {time_taken:.2f}s")
    
    # Performance validation
    if total_time < 300:  # 5 minutes
        logger.info(f"\n✅ EXCELLENT: Completed in under 5 minutes")
    elif total_time < 600:  # 10 minutes
        logger.info(f"\n✅ GOOD: Completed in under 10 minutes")
    else:
        logger.info(f"\n⚠️  SLOW: Took more than 10 minutes")
    
    logger.info(f"\n{'='*80}")
    logger.info("🎉 BASELINE PERFORMANCE TEST COMPLETED!")
    logger.info(f"Results saved to: {log_file}")
    logger.info(f"{'='*80}")
    
    return results

if __name__ == "__main__":
    try:
        results = run_baseline_performance_test()
        print("\n✅ Baseline performance test completed successfully!")
        print("Check the log file for detailed results.")
    except Exception as e:
        print(f"\n❌ Baseline performance test failed: {e}")
        exit(1)
