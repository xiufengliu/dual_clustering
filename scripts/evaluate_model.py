#!/usr/bin/env python3
"""Script to evaluate the neutrosophic forecasting model."""

import argparse
import sys
from pathlib import Path
import json
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.framework.forecasting_framework import NeutrosophicForecastingFramework
from src.evaluation.metrics import ForecastingMetrics
from src.utils.logger import setup_logger


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate neutrosophic forecasting model")
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--dataset", type=str, choices=["solar", "wind"], 
                       default="solar", help="Dataset type")
    parser.add_argument("--horizon", type=int, default=30,
                       help="Forecast horizon")
    parser.add_argument("--confidence", type=float, default=0.95,
                       help="Confidence level for intervals")
    parser.add_argument("--output", type=str, default="results/evaluation",
                       help="Output directory for results")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                       help="Start date for test data")
    parser.add_argument("--end-date", type=str, default="2023-10-03",
                       help="End date for test data")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("evaluate_model")
    logger.info(f"Starting model evaluation")
    
    try:
        # Load trained model
        logger.info(f"Loading model from {args.model}")
        framework = NeutrosophicForecastingFramework.load_model(args.model)
        
        # Load test data
        logger.info("Loading test data...")
        test_data = framework.load_data(
            dataset_type=args.dataset,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Make predictions
        logger.info(f"Making predictions with horizon {args.horizon}")
        predictions = framework.predict(
            data=test_data.iloc[:1],  # Use first point as starting point
            horizon=min(args.horizon, len(test_data)),
            return_intervals=True,
            confidence_level=args.confidence
        )
        
        # Prepare true values for evaluation
        n_pred = len(predictions['predictions'])
        true_values = test_data['energy_generation'].values[:n_pred]
        
        # Calculate metrics
        logger.info("Calculating evaluation metrics...")
        metrics_calculator = ForecastingMetrics()
        
        # Point metrics
        point_metrics = metrics_calculator.calculate_point_metrics(
            true_values, predictions['predictions']
        )
        
        # Interval metrics
        interval_metrics = metrics_calculator.calculate_interval_metrics(
            true_values,
            predictions['lower_bounds'],
            predictions['upper_bounds'],
            args.confidence
        )
        
        # Comprehensive metrics
        comprehensive_metrics = metrics_calculator.calculate_comprehensive_metrics(
            true_values,
            predictions['predictions'],
            predictions['lower_bounds'],
            predictions['upper_bounds'],
            args.confidence
        )
        
        # Compile results
        evaluation_results = {
            'model_path': args.model,
            'dataset_type': args.dataset,
            'horizon': args.horizon,
            'confidence_level': args.confidence,
            'test_period': {
                'start_date': args.start_date,
                'end_date': args.end_date,
                'n_samples': len(test_data),
                'n_predictions': n_pred
            },
            'point_metrics': point_metrics,
            'interval_metrics': interval_metrics,
            'comprehensive_metrics': comprehensive_metrics,
            'predictions_sample': {
                'true_values': true_values[:10].tolist(),
                'predictions': predictions['predictions'][:10].tolist(),
                'lower_bounds': predictions['lower_bounds'][:10].tolist(),
                'upper_bounds': predictions['upper_bounds'][:10].tolist()
            }
        }
        
        # Save results
        output_path = Path(args.output) / f"evaluation_{args.dataset}_{args.horizon}h.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Print key results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Dataset: {args.dataset}")
        print(f"Horizon: {args.horizon}")
        print(f"Test samples: {len(test_data)}")
        print(f"Predictions: {n_pred}")
        print()
        
        print("POINT FORECASTING METRICS:")
        print(f"  RMSE:  {point_metrics['rmse']:.4f}")
        print(f"  MAE:   {point_metrics['mae']:.4f}")
        print(f"  MAPE:  {point_metrics['mape']:.2f}%")
        print(f"  R²:    {point_metrics['r2']:.4f}")
        print(f"  NRMSE: {point_metrics['nrmse']:.4f}")
        print()
        
        print(f"PREDICTION INTERVAL METRICS ({args.confidence:.0%} confidence):")
        print(f"  PICP:  {interval_metrics['picp']:.4f} (target: {args.confidence:.3f})")
        print(f"  PINAW: {interval_metrics['pinaw']:.4f}")
        print(f"  ACE:   {interval_metrics['ace']:.4f}")
        print(f"  CWC:   {interval_metrics['cwc']:.4f}")
        print()
        
        print("ADDITIONAL METRICS:")
        print(f"  Directional Accuracy: {comprehensive_metrics['directional_accuracy']:.4f}")
        print(f"  Correlation: {comprehensive_metrics['summary']['correlation']:.4f}")
        print()
        
        print(f"Results saved to: {output_path}")
        print("="*60)
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        print(f"✗ Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()