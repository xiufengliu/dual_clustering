"""Main experiment runner for the neutrosophic forecasting framework."""

import argparse
import sys
from pathlib import Path
import logging
import json
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.framework.forecasting_framework import NeutrosophicForecastingFramework
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger
from src.evaluation.metrics import ForecastingMetrics


def run_single_experiment(config_name: str, dataset_type: str = "solar", 
                         output_dir: str = "results/experiments") -> dict:
    """Run a single experiment with specified configuration.
    
    Args:
        config_name: Name of configuration to use
        dataset_type: Type of dataset ('solar' or 'wind')
        output_dir: Directory to save results
        
    Returns:
        Dictionary with experiment results
    """
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(output_dir) / f"experiment_{config_name}_{dataset_type}_{timestamp}.log"
    logger = setup_logger("experiment", log_file=log_file)
    
    logger.info(f"Starting experiment: {config_name} on {dataset_type} data")
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.get_config(dataset_type=dataset_type, experiment_name=config_name)
        
        # Initialize framework
        framework = NeutrosophicForecastingFramework(config=config)
        
        # Load data
        data_config = config.get('data', {})
        start_date = data_config.get('start_date', '2019-01-01')
        end_date = data_config.get('end_date', '2023-10-03')
        country = data_config.get('country', 'Denmark')
        
        logger.info(f"Loading {dataset_type} data from {start_date} to {end_date}")
        data = framework.load_data(
            dataset_type=dataset_type,
            start_date=start_date,
            end_date=end_date,
            country=country
        )
        
        # Split data
        train_ratio = config.get('data', {}).get('train_split', 0.8)
        split_idx = int(len(data) * train_ratio)
        
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        logger.info(f"Data split - Train: {len(train_data)}, Test: {len(test_data)}")
        
        # Train framework
        start_time = time.time()
        framework.fit(train_data)
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Make predictions
        horizon = config.get('forecasting', {}).get('horizon', 180)
        confidence_levels = config.get('forecasting', {}).get('confidence_levels', [0.95])
        
        predictions_results = {}
        evaluation_results = {}
        
        for confidence_level in confidence_levels:
            logger.info(f"Making predictions with {confidence_level} confidence level")
            
            # Predict on test data (simplified - using first test point)
            pred_results = framework.predict(
                data=test_data.iloc[:1],
                horizon=min(horizon, len(test_data)),
                return_intervals=True,
                confidence_level=confidence_level
            )
            
            predictions_results[f'confidence_{confidence_level}'] = pred_results
            
            # Evaluate predictions
            if len(pred_results['predictions']) <= len(test_data):
                true_values = test_data['energy_generation'].values[:len(pred_results['predictions'])]
                
                metrics_calculator = ForecastingMetrics()
                
                # Point metrics
                point_metrics = metrics_calculator.calculate_point_metrics(
                    true_values, pred_results['predictions']
                )
                
                # Interval metrics
                interval_metrics = metrics_calculator.calculate_interval_metrics(
                    true_values,
                    pred_results['lower_bounds'],
                    pred_results['upper_bounds'],
                    confidence_level
                )
                
                evaluation_results[f'confidence_{confidence_level}'] = {
                    'point_metrics': point_metrics,
                    'interval_metrics': interval_metrics
                }
        
        # Get feature importance
        feature_importance = framework.get_feature_importance()
        
        # Get framework info
        framework_info = framework.get_framework_info()
        
        # Compile results
        experiment_results = {
            'experiment_info': {
                'config_name': config_name,
                'dataset_type': dataset_type,
                'timestamp': timestamp,
                'training_time': training_time,
                'data_info': {
                    'total_samples': len(data),
                    'train_samples': len(train_data),
                    'test_samples': len(test_data),
                    'start_date': start_date,
                    'end_date': end_date
                }
            },
            'config': config,
            'predictions': predictions_results,
            'evaluation': evaluation_results,
            'feature_importance': feature_importance,
            'framework_info': framework_info
        }
        
        # Save results
        output_path = Path(output_dir) / f"results_{config_name}_{dataset_type}_{timestamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(experiment_results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        
        # Save model
        model_path = Path(output_dir) / f"model_{config_name}_{dataset_type}_{timestamp}.pkl"
        framework.save_model(str(model_path))
        
        logger.info("Experiment completed successfully")
        return experiment_results
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise


def run_multiple_experiments(config_names: list, dataset_types: list = ["solar", "wind"],
                           output_dir: str = "results/experiments") -> dict:
    """Run multiple experiments with different configurations.
    
    Args:
        config_names: List of configuration names
        dataset_types: List of dataset types
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all experiment results
    """
    all_results = {}
    
    for config_name in config_names:
        for dataset_type in dataset_types:
            experiment_key = f"{config_name}_{dataset_type}"
            
            try:
                results = run_single_experiment(config_name, dataset_type, output_dir)
                all_results[experiment_key] = results
                print(f"✓ Completed: {experiment_key}")
                
            except Exception as e:
                print(f"✗ Failed: {experiment_key} - {str(e)}")
                all_results[experiment_key] = {"error": str(e)}
    
    return all_results


def main():
    """Main function for running experiments."""
    parser = argparse.ArgumentParser(description="Run neutrosophic forecasting experiments")
    
    parser.add_argument("--config", type=str, default="base_config",
                       help="Configuration name to use")
    parser.add_argument("--dataset", type=str, choices=["solar", "wind"], default="solar",
                       help="Dataset type")
    parser.add_argument("--output-dir", type=str, default="results/experiments",
                       help="Output directory for results")
    parser.add_argument("--multiple", action="store_true",
                       help="Run multiple experiments")
    parser.add_argument("--configs", nargs="+", default=["base_config"],
                       help="List of configurations for multiple experiments")
    parser.add_argument("--datasets", nargs="+", choices=["solar", "wind"], 
                       default=["solar", "wind"],
                       help="List of datasets for multiple experiments")
    
    args = parser.parse_args()
    
    if args.multiple:
        print("Running multiple experiments...")
        results = run_multiple_experiments(
            config_names=args.configs,
            dataset_types=args.datasets,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("\nExperiment Summary:")
        for key, result in results.items():
            if "error" in result:
                print(f"  {key}: FAILED - {result['error']}")
            else:
                # Extract key metrics
                try:
                    point_metrics = result['evaluation']['confidence_0.95']['point_metrics']
                    rmse = point_metrics['rmse']
                    mae = point_metrics['mae']
                    print(f"  {key}: SUCCESS - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                except:
                    print(f"  {key}: SUCCESS (metrics not available)")
    
    else:
        print(f"Running single experiment: {args.config} on {args.dataset}")
        results = run_single_experiment(
            config_name=args.config,
            dataset_type=args.dataset,
            output_dir=args.output_dir
        )
        
        # Print key results
        try:
            point_metrics = results['evaluation']['confidence_0.95']['point_metrics']
            interval_metrics = results['evaluation']['confidence_0.95']['interval_metrics']
            
            print("\nKey Results:")
            print(f"  RMSE: {point_metrics['rmse']:.4f}")
            print(f"  MAE: {point_metrics['mae']:.4f}")
            print(f"  MAPE: {point_metrics['mape']:.2f}%")
            print(f"  R²: {point_metrics['r2']:.4f}")
            print(f"  PICP: {interval_metrics['picp']:.4f}")
            print(f"  PINAW: {interval_metrics['pinaw']:.4f}")
            
        except Exception as e:
            print(f"Could not extract metrics: {e}")


if __name__ == "__main__":
    main()