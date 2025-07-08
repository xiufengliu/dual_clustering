"""Comprehensive benchmark experiment for comparing forecasting models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import json
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.framework.forecasting_framework import NeutrosophicForecastingFramework
from src.models.baseline_models import BaselineForecasters
from src.evaluation.metrics import ForecastingMetrics
from src.evaluation.statistical_tests import StatisticalTests
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger
from src.utils.math_utils import set_random_seeds

logger = logging.getLogger(__name__)


class BenchmarkExperiment:
    """Comprehensive benchmark experiment for forecasting model comparison."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "results/benchmarks"):
        """Initialize benchmark experiment.
        
        Args:
            config: Experiment configuration
            output_dir: Directory to save results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metrics_calculator = ForecastingMetrics()
        self.statistical_tests = StatisticalTests(alpha=config.get('alpha', 0.05))
        
        # Results storage
        self.results = {
            'experiment_info': {},
            'model_results': {},
            'statistical_tests': {},
            'summary': {}
        }
        
    def prepare_data(self, dataset_type: str, **data_kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare train/validation/test splits.
        
        Args:
            dataset_type: Type of dataset ('solar' or 'wind')
            **data_kwargs: Additional data loading arguments
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info(f"Preparing {dataset_type} data")
        
        # Load data using the framework
        framework = NeutrosophicForecastingFramework()
        data = framework.load_data(dataset_type=dataset_type, **data_kwargs)
        
        # Split data
        train_split = self.config.get('train_split', 0.7)
        val_split = self.config.get('val_split', 0.1)
        
        n_total = len(data)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_data = data.iloc[:n_train]
        val_data = data.iloc[n_train:n_train + n_val]
        test_data = data.iloc[n_train + n_val:]
        
        logger.info(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def create_features(self, data: pd.DataFrame, method: str = 'lag') -> Tuple[np.ndarray, np.ndarray]:
        """Create features for baseline models.
        
        Args:
            data: Input data
            method: Feature creation method
            
        Returns:
            Tuple of (X, y) arrays
        """
        if method == 'lag':
            # Create lag features
            n_lags = self.config.get('n_lags', 24)
            
            # Create lagged features
            X_list = []
            y_list = []
            
            for i in range(n_lags, len(data)):
                X_list.append(data['energy_generation'].iloc[i-n_lags:i].values)
                y_list.append(data['energy_generation'].iloc[i])
                
            X = np.array(X_list)
            y = np.array(y_list)
            
        elif method == 'simple':
            # Simple approach: use previous value as feature
            X = data['energy_generation'].values[:-1].reshape(-1, 1)
            y = data['energy_generation'].values[1:]
            
        else:
            raise ValueError(f"Unknown feature method: {method}")
            
        return X, y
    
    def run_baseline_models(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Run all baseline models.
        
        Args:
            train_data: Training data
            test_data: Test data
            
        Returns:
            Dictionary with baseline model results
        """
        logger.info("Running baseline models")
        
        # Create features
        X_train, y_train = self.create_features(train_data)
        X_test, y_test = self.create_features(test_data)
        
        # Get baseline models
        baseline_models = BaselineForecasters.create_all_models(
            random_state=self.config.get('random_state', 42)
        )
        
        baseline_results = {}
        
        for model_name, model in baseline_models.items():
            logger.info(f"Training {model_name}")
            
            try:
                # Train model
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Make predictions
                start_time = time.time()
                predictions, uncertainties = model.predict_with_uncertainty(X_test)
                prediction_time = time.time() - start_time
                
                # Calculate metrics
                point_metrics = self.metrics_calculator.calculate_point_metrics(y_test, predictions)
                
                # Calculate prediction intervals (simple approach)
                alpha = 1 - self.config.get('confidence_level', 0.95)
                z_score = 1.96  # For 95% confidence
                lower_bounds = predictions - z_score * uncertainties
                upper_bounds = predictions + z_score * uncertainties
                
                interval_metrics = self.metrics_calculator.calculate_interval_metrics(
                    y_test, lower_bounds, upper_bounds, self.config.get('confidence_level', 0.95)
                )
                
                baseline_results[model_name] = {
                    'predictions': predictions,
                    'uncertainties': uncertainties,
                    'lower_bounds': lower_bounds,
                    'upper_bounds': upper_bounds,
                    'point_metrics': point_metrics,
                    'interval_metrics': interval_metrics,
                    'training_time': training_time,
                    'prediction_time': prediction_time,
                    'errors': y_test - predictions
                }
                
                logger.info(f"{model_name} - RMSE: {point_metrics['rmse']:.4f}, MAE: {point_metrics['mae']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to run {model_name}: {e}")
                baseline_results[model_name] = {
                    'error': str(e),
                    'failed': True
                }
        
        return baseline_results
    
    def run_proposed_model(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Run the proposed neutrosophic model.
        
        Args:
            train_data: Training data
            test_data: Test data
            
        Returns:
            Dictionary with proposed model results
        """
        logger.info("Running proposed neutrosophic model")
        
        try:
            # Initialize framework
            framework = NeutrosophicForecastingFramework(config=self.config)
            
            # Train model
            start_time = time.time()
            framework.fit(train_data)
            training_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            predictions_dict = framework.predict(
                test_data.iloc[:1],  # Starting point
                horizon=len(test_data),
                return_intervals=True,
                confidence_level=self.config.get('confidence_level', 0.95)
            )
            prediction_time = time.time() - start_time
            
            predictions = predictions_dict['predictions']
            lower_bounds = predictions_dict['lower_bounds']
            upper_bounds = predictions_dict['upper_bounds']
            
            # Get true values
            y_test = test_data['energy_generation'].values[:len(predictions)]
            
            # Calculate metrics
            point_metrics = self.metrics_calculator.calculate_point_metrics(y_test, predictions)
            interval_metrics = self.metrics_calculator.calculate_interval_metrics(
                y_test, lower_bounds, upper_bounds, self.config.get('confidence_level', 0.95)
            )
            
            # Get feature importance
            feature_importance = framework.get_feature_importance()
            
            proposed_results = {
                'predictions': predictions,
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'point_metrics': point_metrics,
                'interval_metrics': interval_metrics,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'errors': y_test - predictions,
                'feature_importance': feature_importance
            }
            
            logger.info(f"Proposed model - RMSE: {point_metrics['rmse']:.4f}, MAE: {point_metrics['mae']:.4f}")
            
            return proposed_results
            
        except Exception as e:
            logger.error(f"Failed to run proposed model: {e}")
            return {
                'error': str(e),
                'failed': True
            }
    
    def run_statistical_tests(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run statistical significance tests.
        
        Args:
            model_results: Results from all models
            
        Returns:
            Dictionary with statistical test results
        """
        logger.info("Running statistical significance tests")
        
        # Extract errors for models that succeeded
        errors_dict = {}
        for model_name, results in model_results.items():
            if 'errors' in results and not results.get('failed', False):
                errors_dict[model_name] = results['errors']
        
        if len(errors_dict) < 2:
            logger.warning("Not enough models for statistical testing")
            return {}
        
        statistical_results = {}
        
        # Comprehensive model comparison
        if 'neutrosophic' in errors_dict:
            comprehensive_results = self.statistical_tests.comprehensive_model_comparison(
                errors_dict, reference_model='neutrosophic'
            )
            statistical_results['comprehensive_comparison'] = comprehensive_results
        
        # Pairwise DM tests against proposed model
        if 'neutrosophic' in errors_dict:
            pairwise_tests = {}
            for model_name in errors_dict:
                if model_name != 'neutrosophic':
                    dm_result = self.statistical_tests.modified_diebold_mariano_test(
                        errors_dict['neutrosophic'], errors_dict[model_name]
                    )
                    pairwise_tests[f"neutrosophic_vs_{model_name}"] = dm_result
            
            statistical_results['pairwise_dm_tests'] = pairwise_tests
        
        return statistical_results

    def run_experiment(self, dataset_type: str, **data_kwargs) -> Dict[str, Any]:
        """Run complete benchmark experiment.

        Args:
            dataset_type: Type of dataset ('solar' or 'wind')
            **data_kwargs: Additional data loading arguments

        Returns:
            Complete experiment results
        """
        logger.info(f"Starting benchmark experiment for {dataset_type}")

        # Set random seeds for reproducibility
        set_random_seeds(self.config.get('random_state', 42))

        # Record experiment info
        self.results['experiment_info'] = {
            'dataset_type': dataset_type,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'data_kwargs': data_kwargs
        }

        # Prepare data
        train_data, val_data, test_data = self.prepare_data(dataset_type, **data_kwargs)

        self.results['experiment_info']['data_info'] = {
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'total_samples': len(train_data) + len(val_data) + len(test_data)
        }

        # Run baseline models
        baseline_results = self.run_baseline_models(train_data, test_data)

        # Run proposed model
        proposed_results = self.run_proposed_model(train_data, test_data)

        # Combine all model results
        all_model_results = {**baseline_results, 'neutrosophic': proposed_results}
        self.results['model_results'] = all_model_results

        # Run statistical tests
        statistical_results = self.run_statistical_tests(all_model_results)
        self.results['statistical_tests'] = statistical_results

        # Generate summary
        self.results['summary'] = self.generate_summary()

        # Save results
        self.save_results(dataset_type)

        logger.info("Benchmark experiment completed")
        return self.results

    def generate_summary(self) -> Dict[str, Any]:
        """Generate experiment summary."""
        summary = {
            'model_rankings': {},
            'best_models': {},
            'significant_improvements': []
        }

        # Extract point metrics for ranking
        model_metrics = {}
        for model_name, results in self.results['model_results'].items():
            if 'point_metrics' in results and not results.get('failed', False):
                model_metrics[model_name] = results['point_metrics']

        # Rank models by different metrics
        metrics_to_rank = ['rmse', 'mae', 'mape']
        for metric in metrics_to_rank:
            if model_metrics:
                sorted_models = sorted(
                    model_metrics.items(),
                    key=lambda x: x[1].get(metric, float('inf'))
                )
                summary['model_rankings'][metric] = [
                    {'model': name, 'value': metrics[metric]}
                    for name, metrics in sorted_models
                ]
                summary['best_models'][metric] = sorted_models[0][0]

        # Extract significant improvements
        if 'pairwise_dm_tests' in self.results['statistical_tests']:
            for test_name, test_result in self.results['statistical_tests']['pairwise_dm_tests'].items():
                if test_result.get('significant', False):
                    summary['significant_improvements'].append({
                        'comparison': test_name,
                        'p_value': test_result['p_value'],
                        'statistic': test_result['statistic']
                    })

        return summary

    def save_results(self, dataset_type: str):
        """Save experiment results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{dataset_type}_{timestamp}.json"
        filepath = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(self.results)

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {filepath}")

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif pd.isna(obj):
            return None
        else:
            return obj


def main():
    """Main function for running benchmark experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive benchmark experiment")
    parser.add_argument("--dataset", type=str, choices=["solar", "wind"], default="solar",
                       help="Dataset type")
    parser.add_argument("--config", type=str, default="benchmark_config",
                       help="Configuration name")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks",
                       help="Output directory")
    parser.add_argument("--start-date", type=str, default="2019-01-01",
                       help="Start date for data")
    parser.add_argument("--end-date", type=str, default="2023-10-03",
                       help="End date for data")

    args = parser.parse_args()

    # Setup logging
    log_file = Path(args.output_dir) / f"benchmark_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger("benchmark", log_file=log_file)

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config(dataset_type=args.dataset, experiment_name=args.config)

    # Run experiment
    experiment = BenchmarkExperiment(config, args.output_dir)
    results = experiment.run_experiment(
        dataset_type=args.dataset,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK EXPERIMENT SUMMARY")
    print("="*50)

    summary = results['summary']

    # Print model rankings
    for metric in ['rmse', 'mae']:
        if metric in summary['model_rankings']:
            print(f"\n{metric.upper()} Rankings:")
            for i, model_info in enumerate(summary['model_rankings'][metric][:5]):
                print(f"  {i+1}. {model_info['model']}: {model_info['value']:.4f}")

    # Print significant improvements
    if summary['significant_improvements']:
        print(f"\nSignificant Improvements (α=0.05):")
        for improvement in summary['significant_improvements']:
            print(f"  {improvement['comparison']}: p={improvement['p_value']:.4f}")

    print("\n" + "="*50)


if __name__ == "__main__":
    main()
