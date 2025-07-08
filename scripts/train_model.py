#!/usr/bin/env python3
"""Script to train the neutrosophic forecasting model."""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.framework.forecasting_framework import NeutrosophicForecastingFramework
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train neutrosophic forecasting model")
    
    parser.add_argument("--dataset", type=str, choices=["solar", "wind"], 
                       default="solar", help="Dataset type")
    parser.add_argument("--config", type=str, default="base_config",
                       help="Configuration name")
    parser.add_argument("--output", type=str, default="results/models",
                       help="Output directory for trained model")
    parser.add_argument("--start-date", type=str, default="2019-01-01",
                       help="Start date for data")
    parser.add_argument("--end-date", type=str, default="2023-10-03",
                       help="End date for data")
    parser.add_argument("--country", type=str, default="Denmark",
                       help="Country for data")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("train_model")
    logger.info(f"Starting model training for {args.dataset} data")
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.get_config(dataset_type=args.dataset)
        
        # Override data parameters
        config['data'].update({
            'start_date': args.start_date,
            'end_date': args.end_date,
            'country': args.country
        })
        
        # Initialize framework
        framework = NeutrosophicForecastingFramework(config=config)
        
        # Load and prepare data
        logger.info("Loading data...")
        data = framework.load_data(
            dataset_type=args.dataset,
            start_date=args.start_date,
            end_date=args.end_date,
            country=args.country
        )
        
        # Train model
        logger.info("Training model...")
        framework.fit(data)
        
        # Save model
        output_path = Path(args.output) / f"{args.dataset}_model_{args.config}.pkl"
        framework.save_model(str(output_path))
        
        logger.info(f"Model saved to {output_path}")
        
        # Print model info
        info = framework.get_framework_info()
        logger.info(f"Model trained with {info['n_features']} features")
        
        # Print feature importance
        feature_importance = framework.get_feature_importance()
        logger.info("Top 5 most important features:")
        for i, (name, importance) in enumerate(feature_importance['feature_importance_ranking'][:5]):
            logger.info(f"  {i+1}. {name}: {importance:.4f}")
        
        print(f"✓ Training completed successfully. Model saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        print(f"✗ Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()