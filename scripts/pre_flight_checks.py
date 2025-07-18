#!/usr/bin/env python3
"""
Comprehensive Pre-Flight Validation System
Systematic checks before running dual clustering experiments
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from src.utils.config_manager import ConfigManager
    from src.data.preprocessor import DataPreprocessor
except ImportError:
    # Fallback for different import paths
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils.config_manager import ConfigManager
    from src.data.preprocessor import DataPreprocessor

class PreFlightValidator:
    """Comprehensive pre-flight validation system"""
    
    def __init__(self, config_name: str = 'benchmark_config'):
        self.config_name = config_name
        self.config = ConfigManager().get_config(config_name)
        self.validation_results = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path('results/validation')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'pre_flight_validation_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def validate_environment(self) -> bool:
        """Validate Python environment and dependencies"""
        self.logger.info("=== Environment Validation ===")
        
        try:
            # Python version check
            python_version = sys.version_info
            self.logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            if python_version < (3, 8):
                self.logger.error("Python 3.8+ required")
                return False
            
            # Core dependencies check
            required_packages = {
                'numpy': '1.21.0',
                'pandas': '1.3.0', 
                'scikit-learn': '1.0.0',
                'scipy': '1.7.0',
                'matplotlib': '3.5.0',
                'seaborn': '0.11.0',
                'pyyaml': '6.0',
                'tqdm': '4.62.0'
            }
            
            for package, min_version in required_packages.items():
                try:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                    self.logger.info(f"{package}: {version}")
                except ImportError as e:
                    self.logger.error(f"Missing package: {package}")
                    return False
            
            # GPU availability check
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    self.logger.info("GPU available")
                else:
                    self.logger.warning("GPU not available")
            except FileNotFoundError:
                self.logger.warning("nvidia-smi not found")
            
            self.validation_results['environment'] = True
            return True
            
        except Exception as e:
            self.logger.error(f"Environment validation failed: {e}")
            self.validation_results['environment'] = False
            return False
    
    def validate_data_integrity(self) -> bool:
        """Validate data integrity and consistency"""
        self.logger.info("=== Data Integrity Validation ===")
        
        try:
            data_dir = Path('data/processed')
            if not data_dir.exists():
                self.logger.error("Processed data directory not found")
                return False
            
            dataset_files = list(data_dir.glob('*.csv'))
            if not dataset_files:
                self.logger.error("No processed datasets found")
                return False
            
            self.logger.info(f"Found {len(dataset_files)} datasets")
            
            # Validate each dataset
            dataset_info = {}
            for dataset_file in dataset_files:
                try:
                    df = pd.read_csv(dataset_file)
                    dataset_name = dataset_file.stem
                    
                    # Basic validation
                    info = {
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'dtypes': df.dtypes.to_dict(),
                        'missing_values': df.isnull().sum().to_dict(),
                        'energy_generation_stats': {}
                    }
                    
                    # Energy generation column validation
                    if 'energy_generation' in df.columns:
                        energy_col = df['energy_generation']
                        info['energy_generation_stats'] = {
                            'count': len(energy_col),
                            'mean': float(energy_col.mean()),
                            'std': float(energy_col.std()),
                            'min': float(energy_col.min()),
                            'max': float(energy_col.max()),
                            'null_count': int(energy_col.isnull().sum())
                        }
                        
                        # Check for anomalies
                        if energy_col.isnull().sum() > 0:
                            self.logger.warning(f"{dataset_name}: {energy_col.isnull().sum()} null values in energy_generation")
                        
                        if energy_col.min() < 0:
                            self.logger.warning(f"{dataset_name}: Negative energy values detected")
                    else:
                        self.logger.error(f"{dataset_name}: Missing energy_generation column")
                        return False
                    
                    dataset_info[dataset_name] = info
                    self.logger.info(f"{dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                    
                except Exception as e:
                    self.logger.error(f"Failed to validate {dataset_file}: {e}")
                    return False
            
            # Save dataset info
            info_file = Path('results/validation/dataset_info.json')
            info_file.parent.mkdir(parents=True, exist_ok=True)
            with open(info_file, 'w') as f:
                json.dump(dataset_info, f, indent=2, default=str)
            
            self.validation_results['data_integrity'] = True
            return True
            
        except Exception as e:
            self.logger.error(f"Data integrity validation failed: {e}")
            self.validation_results['data_integrity'] = False
            return False
    
    def validate_configuration(self) -> bool:
        """Validate configuration files and parameters"""
        self.logger.info("=== Configuration Validation ===")
        
        try:
            # Check configuration structure
            required_sections = ['clustering', 'neutrosophic', 'random_forest']
            for section in required_sections:
                if section not in self.config:
                    self.logger.error(f"Missing configuration section: {section}")
                    return False
            
            # Validate parameter types and ranges
            clustering_config = self.config['clustering']
            if not isinstance(clustering_config.get('n_clusters'), int) or clustering_config.get('n_clusters') < 2:
                self.logger.error("Invalid n_clusters parameter")
                return False
            
            neutrosophic_config = self.config['neutrosophic']
            entropy_epsilon = neutrosophic_config.get('entropy_epsilon')
            if not isinstance(entropy_epsilon, (int, float)) or entropy_epsilon <= 0:
                self.logger.error("Invalid entropy_epsilon parameter")
                return False
            
            self.logger.info("Configuration validation passed")
            self.validation_results['configuration'] = True
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            self.validation_results['configuration'] = False
            return False
    
    def estimate_resource_requirements(self) -> bool:
        """Estimate memory and computational requirements"""
        self.logger.info("=== Resource Requirements Estimation ===")
        
        try:
            # Get system resources
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
            
            self.logger.info(f"Available memory: {memory_gb:.1f} GB")
            self.logger.info(f"CPU cores: {cpu_count}")
            
            # Estimate memory requirements based on dataset sizes
            data_dir = Path('data/processed')
            total_data_size = 0
            max_dataset_size = 0
            
            for dataset_file in data_dir.glob('*.csv'):
                size_mb = dataset_file.stat().st_size / (1024**2)
                total_data_size += size_mb
                max_dataset_size = max(max_dataset_size, size_mb)
            
            # Rough estimation: 10x data size for processing
            estimated_memory_gb = (max_dataset_size * 10) / 1024
            
            self.logger.info(f"Total data size: {total_data_size:.1f} MB")
            self.logger.info(f"Largest dataset: {max_dataset_size:.1f} MB")
            self.logger.info(f"Estimated memory requirement: {estimated_memory_gb:.1f} GB")
            
            if estimated_memory_gb > memory_gb * 0.8:
                self.logger.warning("High memory usage expected - consider reducing dataset size")
            
            self.validation_results['resource_estimation'] = {
                'available_memory_gb': memory_gb,
                'estimated_memory_gb': estimated_memory_gb,
                'cpu_cores': cpu_count,
                'memory_sufficient': estimated_memory_gb <= memory_gb * 0.8
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Resource estimation failed: {e}")
            self.validation_results['resource_estimation'] = False
            return False
    
    def run_component_tests(self) -> bool:
        """Run basic component functionality tests"""
        self.logger.info("=== Component Testing ===")
        
        try:
            # Test data preprocessor
            preprocessor = DataPreprocessor()
            
            # Create synthetic test data
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
                'energy_generation': np.random.normal(50, 10, 100)
            })
            
            processed_data = preprocessor.preprocess(test_data)
            
            if processed_data is None or len(processed_data) == 0:
                self.logger.error("Data preprocessor test failed")
                return False
            
            self.logger.info("Component tests passed")
            self.validation_results['component_tests'] = True
            return True
            
        except Exception as e:
            self.logger.error(f"Component testing failed: {e}")
            self.validation_results['component_tests'] = False
            return False
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config_name': self.config_name,
            'validation_results': self.validation_results,
            'overall_status': all(self.validation_results.values()),
            'recommendations': []
        }
        
        # Generate recommendations
        if not self.validation_results.get('environment', True):
            report['recommendations'].append("Fix environment issues before proceeding")
        
        if not self.validation_results.get('data_integrity', True):
            report['recommendations'].append("Resolve data integrity issues")
        
        resource_info = self.validation_results.get('resource_estimation', {})
        if isinstance(resource_info, dict) and not resource_info.get('memory_sufficient', True):
            report['recommendations'].append("Consider running on a machine with more memory")
        
        return report
    
    def run_all_validations(self) -> bool:
        """Run all validation checks"""
        self.logger.info("Starting comprehensive pre-flight validation")
        
        validations = [
            self.validate_environment,
            self.validate_data_integrity,
            self.validate_configuration,
            self.estimate_resource_requirements,
            self.run_component_tests
        ]
        
        all_passed = True
        for validation in validations:
            try:
                if not validation():
                    all_passed = False
            except Exception as e:
                self.logger.error(f"Validation failed with exception: {e}")
                all_passed = False
        
        # Generate and save report
        report = self.generate_validation_report()
        report_file = Path('results/validation/pre_flight_report.json')
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        if all_passed:
            self.logger.info("✅ All pre-flight validations passed")
        else:
            self.logger.error("❌ Some pre-flight validations failed")
        
        return all_passed

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-flight validation for dual clustering experiments')
    parser.add_argument('--config', default='benchmark_config', help='Configuration name')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = PreFlightValidator(args.config)
    success = validator.run_all_validations()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
