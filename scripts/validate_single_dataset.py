#!/usr/bin/env python3
"""
Single Dataset Validation Script
Comprehensive validation for individual datasets before full experiments
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from framework.forecasting_framework import NeutrosophicForecastingFramework
from utils.math_utils import set_random_seeds
from utils.config_manager import ConfigManager

class SingleDatasetValidator:
    """Comprehensive validation for a single dataset"""
    
    def __init__(self, dataset_name: str, config_name: str = 'benchmark_config'):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.config = ConfigManager().get_config(config_name)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for validation"""
        log_dir = Path('results/validation/single_dataset')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{self.dataset_name}_validation_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_dataset(self) -> Optional[pd.DataFrame]:
        """Load and validate dataset file"""
        self.logger.info(f"Loading dataset: {self.dataset_name}")
        
        try:
            dataset_path = Path(f'data/processed/{self.dataset_name}.csv')
            if not dataset_path.exists():
                self.logger.error(f"Dataset file not found: {dataset_path}")
                return None
            
            data = pd.read_csv(dataset_path)
            self.logger.info(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            return None
    
    def validate_data_structure(self, data: pd.DataFrame) -> bool:
        """Validate basic data structure and requirements"""
        self.logger.info("Validating data structure")
        
        try:
            # Check required columns
            required_columns = ['energy_generation']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check data types
            energy_col = data['energy_generation']
            if not pd.api.types.is_numeric_dtype(energy_col):
                self.logger.error("energy_generation column is not numeric")
                return False
            
            # Check for minimum data size
            if len(data) < 100:
                self.logger.error(f"Dataset too small: {len(data)} rows (minimum: 100)")
                return False
            
            # Check for excessive missing values
            missing_pct = energy_col.isnull().sum() / len(energy_col) * 100
            if missing_pct > 10:
                self.logger.error(f"Too many missing values: {missing_pct:.1f}%")
                return False
            
            self.logger.info("Data structure validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data structure validation failed: {e}")
            return False
    
    def validate_data_quality(self, data: pd.DataFrame) -> bool:
        """Validate data quality and detect anomalies"""
        self.logger.info("Validating data quality")
        
        try:
            energy_col = data['energy_generation'].dropna()
            
            # Statistical validation
            stats = {
                'count': len(energy_col),
                'mean': energy_col.mean(),
                'std': energy_col.std(),
                'min': energy_col.min(),
                'max': energy_col.max(),
                'q25': energy_col.quantile(0.25),
                'q75': energy_col.quantile(0.75)
            }
            
            self.logger.info(f"Energy generation stats: {stats}")
            
            # Check for negative values
            if stats['min'] < 0:
                negative_count = (energy_col < 0).sum()
                self.logger.warning(f"Found {negative_count} negative energy values")
            
            # Check for extreme outliers (beyond 5 standard deviations)
            z_scores = np.abs((energy_col - stats['mean']) / stats['std'])
            extreme_outliers = (z_scores > 5).sum()
            
            if extreme_outliers > 0:
                self.logger.warning(f"Found {extreme_outliers} extreme outliers")
            
            # Check for constant values
            if stats['std'] == 0:
                self.logger.error("Energy generation has zero variance (constant values)")
                return False
            
            # Check for reasonable range
            if stats['max'] / stats['mean'] > 100:  # Max is 100x the mean
                self.logger.warning("Very high maximum value detected - possible data error")
            
            self.logger.info("Data quality validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data quality validation failed: {e}")
            return False
    
    def test_framework_components(self, data: pd.DataFrame) -> bool:
        """Test individual framework components with the dataset"""
        self.logger.info("Testing framework components")
        
        try:
            # Use a subset for testing to save time
            test_data = data.head(min(500, len(data))).copy()
            
            set_random_seeds(42)
            framework = NeutrosophicForecastingFramework()
            
            # Test data preprocessing
            self.logger.info("Testing data preprocessing")
            framework.preprocessor.preprocess(test_data.copy())
            
            # Test dual clustering
            self.logger.info("Testing dual clustering")
            X = test_data['energy_generation'].values.reshape(-1, 1).astype(np.float64)
            framework.dual_clusterer.fit(X)
            
            # Test integrated features
            integrated_features = framework.dual_clusterer.get_integrated_features()
            if integrated_features.dtype != np.float64:
                self.logger.error(f"Integrated features wrong dtype: {integrated_features.dtype}")
                return False
            
            # Test neutrosophic transformation
            self.logger.info("Testing neutrosophic transformation")
            kmeans_labels, fcm_memberships = framework.dual_clusterer.get_cluster_assignments()
            neutrosophic_components = framework.neutrosophic_transformer.transform(
                kmeans_labels, fcm_memberships
            )
            
            # Test enriched features creation
            enriched_features = framework.neutrosophic_transformer.create_enriched_features(
                X, integrated_features, neutrosophic_components
            )
            
            if enriched_features.dtype != np.float64:
                self.logger.error(f"Enriched features wrong dtype: {enriched_features.dtype}")
                return False
            
            self.logger.info("Framework components test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Framework components test failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def test_end_to_end_pipeline(self, data: pd.DataFrame) -> bool:
        """Test complete end-to-end pipeline"""
        self.logger.info("Testing end-to-end pipeline")
        
        try:
            # Use subset for faster testing
            test_data = data.head(min(200, len(data))).copy()
            
            set_random_seeds(42)
            framework = NeutrosophicForecastingFramework()
            
            # Test fitting
            self.logger.info("Testing framework fitting")
            framework.fit(test_data)
            
            # Test prediction
            self.logger.info("Testing prediction")
            predictions = framework.predict(horizon=3)
            
            # Validate predictions
            if 'predictions' not in predictions:
                self.logger.error("Missing predictions in output")
                return False
            
            pred_array = predictions['predictions']
            if not isinstance(pred_array, np.ndarray):
                self.logger.error("Predictions not numpy array")
                return False
            
            if len(pred_array) != 3:
                self.logger.error(f"Wrong prediction length: {len(pred_array)} (expected: 3)")
                return False
            
            if not np.all(np.isfinite(pred_array)):
                self.logger.error("Non-finite values in predictions")
                return False
            
            self.logger.info("End-to-end pipeline test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"End-to-end pipeline test failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def generate_validation_report(self, results: Dict) -> Dict:
        """Generate comprehensive validation report"""
        report = {
            'dataset_name': self.dataset_name,
            'timestamp': datetime.now().isoformat(),
            'config_name': self.config_name,
            'validation_results': results,
            'overall_status': all(results.values()),
            'recommendations': []
        }
        
        # Generate specific recommendations
        if not results.get('data_structure', True):
            report['recommendations'].append("Fix data structure issues")
        
        if not results.get('data_quality', True):
            report['recommendations'].append("Address data quality concerns")
        
        if not results.get('framework_components', True):
            report['recommendations'].append("Debug framework component issues")
        
        if not results.get('end_to_end', True):
            report['recommendations'].append("Fix end-to-end pipeline issues")
        
        if report['overall_status']:
            report['recommendations'].append("Dataset ready for full experiments")
        
        return report
    
    def run_validation(self) -> bool:
        """Run complete validation for the dataset"""
        self.logger.info(f"Starting validation for dataset: {self.dataset_name}")
        
        # Load dataset
        data = self.load_dataset()
        if data is None:
            return False
        
        # Run validation steps
        results = {}
        
        results['data_structure'] = self.validate_data_structure(data)
        results['data_quality'] = self.validate_data_quality(data)
        
        if results['data_structure'] and results['data_quality']:
            results['framework_components'] = self.test_framework_components(data)
            results['end_to_end'] = self.test_end_to_end_pipeline(data)
        else:
            results['framework_components'] = False
            results['end_to_end'] = False
        
        # Generate report
        report = self.generate_validation_report(results)
        
        # Save report
        report_dir = Path('results/validation/single_dataset')
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f'{self.dataset_name}_report.json'
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log final status
        if report['overall_status']:
            self.logger.info(f"✅ Dataset {self.dataset_name} validation passed")
        else:
            self.logger.error(f"❌ Dataset {self.dataset_name} validation failed")
        
        return report['overall_status']

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Single dataset validation')
    parser.add_argument('--dataset', required=True, help='Dataset name (without .csv extension)')
    parser.add_argument('--config', default='benchmark_config', help='Configuration name')
    parser.add_argument('--validate-all', action='store_true', help='Run all validation steps')
    
    args = parser.parse_args()
    
    validator = SingleDatasetValidator(args.dataset, args.config)
    success = validator.run_validation()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
