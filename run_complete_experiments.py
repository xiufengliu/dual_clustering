#!/usr/bin/env python3
"""
Complete experiment runner for TNNLS paper.
Downloads data and runs all experimental components.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.data_loader import ENTSOEDataLoader, CSVDataLoader
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def download_sample_datasets():
    """Download and prepare sample datasets for experiments."""
    logger.info("Downloading and preparing sample datasets...")
    
    # Create data directories
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    for dir_path in [raw_dir, processed_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize data loader
    loader = ENTSOEDataLoader(cache_dir=str(raw_dir))
    
    # Define datasets to create
    datasets = {
        'entso_e_solar': {
            'start_date': '2019-01-01',
            'end_date': '2023-10-03',
            'country': 'Denmark',
            'energy_type': 'solar'
        },
        'entso_e_wind': {
            'start_date': '2019-01-01', 
            'end_date': '2023-10-03',
            'country': 'Spain',
            'energy_type': 'wind_onshore'
        },
        'gefcom2014_solar': {
            'start_date': '2020-01-01',
            'end_date': '2021-12-31',
            'country': 'Australia',
            'energy_type': 'solar'
        },
        'gefcom2014_wind': {
            'start_date': '2020-01-01',
            'end_date': '2022-12-31',
            'country': 'Germany',
            'energy_type': 'wind_onshore'
        },
        'nrel_solar': {
            'start_date': '2021-01-01',
            'end_date': '2023-12-31',
            'country': 'California',
            'energy_type': 'solar'
        },
        'nrel_wind': {
            'start_date': '2021-01-01',
            'end_date': '2023-12-31',
            'country': 'Texas',
            'energy_type': 'wind_onshore'
        }
    }
    
    # Download/generate each dataset
    for dataset_name, config in datasets.items():
        logger.info(f"Preparing dataset: {dataset_name}")
        
        try:
            # Load data (will generate synthetic data since no API token)
            data = loader.load_data(
                start_date=config['start_date'],
                end_date=config['end_date'],
                country=config['country'],
                energy_type=config['energy_type']
            )
            
            # Add some realistic features for experiments
            data = add_features(data, config['energy_type'])
            
            # Save processed data
            output_file = processed_dir / f"{dataset_name}.csv"
            data.to_csv(output_file, index=False)
            
            logger.info(f"Dataset {dataset_name} saved: {len(data)} samples, {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to prepare dataset {dataset_name}: {e}")
    
    logger.info("Dataset preparation completed!")
    return True


def add_features(data: pd.DataFrame, energy_type: str) -> pd.DataFrame:
    """Add realistic features to the dataset."""
    data = data.copy()
    
    # Extract time-based features
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['month'] = data['timestamp'].dt.month
    data['day_of_year'] = data['timestamp'].dt.dayofyear
    
    # Add weather-like features (synthetic)
    np.random.seed(42)  # For reproducibility
    n_points = len(data)
    
    if energy_type == 'solar':
        # Solar-related features
        data['cloud_cover'] = np.random.beta(2, 5, n_points)  # More clear days
        data['temperature'] = 15 + 10 * np.sin((data['day_of_year'] - 80) * 2 * np.pi / 365) + np.random.normal(0, 3, n_points)
        data['humidity'] = np.random.beta(3, 3, n_points)
        
        # Adjust generation based on cloud cover
        data['energy_generation'] *= (1 - 0.7 * data['cloud_cover'])
        
    else:  # wind
        # Wind-related features
        data['wind_speed'] = np.random.gamma(2, 3, n_points)  # Gamma distribution for wind speed
        data['wind_direction'] = np.random.uniform(0, 360, n_points)
        data['pressure'] = 1013 + np.random.normal(0, 10, n_points)
        
        # Adjust generation based on wind speed (simplified power curve)
        wind_power_curve = np.where(data['wind_speed'] < 3, 0,
                           np.where(data['wind_speed'] < 12, 
                                   (data['wind_speed'] - 3) / 9,
                                   np.where(data['wind_speed'] < 25, 1, 0)))
        data['energy_generation'] *= wind_power_curve
    
    # Ensure non-negative generation
    data['energy_generation'] = np.maximum(0, data['energy_generation'])
    
    return data


def install_dependencies():
    """Install required dependencies for experiments."""
    logger.info("Installing required dependencies...")
    
    dependencies = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0"
    ]
    
    optional_dependencies = [
        "statsmodels>=0.13.0",
        "lightgbm>=3.3.0",
        "torch>=1.12.0",
        "plotly>=5.0.0"
    ]
    
    # Install core dependencies
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            logger.info(f"Installed: {dep}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {dep}: {e}")
            return False
    
    # Install optional dependencies (don't fail if these don't work)
    for dep in optional_dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            logger.info(f"Installed optional: {dep}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install optional dependency {dep}: {e}")
    
    logger.info("Dependency installation completed!")
    return True


def run_quick_test():
    """Run a quick test to verify everything is working."""
    logger.info("Running quick test...")
    
    try:
        # Test data loading
        from src.data.data_loader import ENTSOEDataLoader
        loader = ENTSOEDataLoader()
        test_data = loader.load_data('2023-01-01', '2023-01-07', 'Denmark', 'solar')
        logger.info(f"Data loading test passed: {len(test_data)} samples")
        
        # Test basic imports
        from src.framework.forecasting_framework import NeutrosophicForecastingFramework
        from src.models.baseline_models import BaselineForecasters
        from src.evaluation.metrics import ForecastingMetrics
        
        logger.info("Import test passed")
        
        # Test baseline model creation
        models = BaselineForecasters.get_available_models()
        logger.info(f"Available baseline models: {list(models.keys())}")
        
        logger.info("Quick test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        return False


def run_experiments(experiment_type: str = "quick"):
    """Run the experiments."""
    logger.info(f"Running {experiment_type} experiments...")
    
    if experiment_type == "quick":
        # Run a subset of experiments for testing
        cmd = [
            sys.executable, "experiments/comprehensive_evaluation.py",
            "--config", "benchmark_config",
            "--datasets", "entso_e_solar",
            "--skip-computational",
            "--skip-cross-dataset",
            "--skip-robustness"
        ]
    elif experiment_type == "main":
        # Run main comparison experiments only
        cmd = [
            sys.executable, "experiments/comprehensive_evaluation.py",
            "--config", "benchmark_config",
            "--datasets", "entso_e_solar", "entso_e_wind", "gefcom2014_solar",
            "--skip-sensitivity",
            "--skip-computational", 
            "--skip-cross-dataset",
            "--skip-robustness"
        ]
    elif experiment_type == "full":
        # Run all experiments
        cmd = [
            sys.executable, "experiments/comprehensive_evaluation.py",
            "--config", "benchmark_config",
            "--datasets", "entso_e_solar", "entso_e_wind", "gefcom2014_solar"
        ]
    else:
        logger.error(f"Unknown experiment type: {experiment_type}")
        return False
    
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            logger.info("Experiments completed successfully!")
            logger.info("STDOUT:")
            logger.info(result.stdout)
            return True
        else:
            logger.error("Experiments failed!")
            logger.error("STDERR:")
            logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Experiments timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"Failed to run experiments: {e}")
        return False


def generate_visualizations():
    """Generate visualizations from experiment results."""
    logger.info("Generating visualizations...")
    
    try:
        # Find the most recent results file
        results_dir = Path("results/comprehensive")
        if not results_dir.exists():
            logger.error("No results directory found")
            return False
        
        result_files = list(results_dir.glob("comprehensive_evaluation_*.json"))
        if not result_files:
            logger.error("No result files found")
            return False
        
        # Get the most recent file
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using results file: {latest_file}")
        
        # Generate plots
        cmd = [
            sys.executable, "-c",
            f"""
import json
from src.visualization.experiment_plots import ExperimentVisualizer

# Load results
with open('{latest_file}', 'r') as f:
    results = json.load(f)

# Generate plots
visualizer = ExperimentVisualizer()
visualizer.generate_all_plots(results, 'results/figures')
print("Visualizations generated successfully!")
"""
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Visualizations generated successfully!")
            logger.info(result.stdout)
            return True
        else:
            logger.error("Visualization generation failed!")
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
        return False


def main():
    """Main function to run complete experiments."""
    parser = argparse.ArgumentParser(description="Complete experiment runner for TNNLS paper")
    parser.add_argument("--step", choices=["all", "deps", "data", "test", "quick", "main", "full", "viz"],
                       default="all", help="Which step to run")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-data", action="store_true", help="Skip data download")
    parser.add_argument("--skip-test", action="store_true", help="Skip quick test")
    parser.add_argument("--output-dir", default="results", help="Output directory")

    args = parser.parse_args()

    # Setup logging
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"experiment_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = setup_logger("experiment_runner", log_file=log_file, level=logging.INFO)

    logger.info("="*80)
    logger.info("TNNLS PAPER EXPERIMENT RUNNER")
    logger.info("="*80)
    logger.info(f"Step: {args.step}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_file}")

    success = True

    try:
        if args.step in ["all", "deps"] and not args.skip_deps:
            logger.info("\n" + "="*50)
            logger.info("STEP 1: Installing Dependencies")
            logger.info("="*50)
            if not install_dependencies():
                logger.error("Dependency installation failed")
                success = False

        if success and args.step in ["all", "data"] and not args.skip_data:
            logger.info("\n" + "="*50)
            logger.info("STEP 2: Downloading/Preparing Data")
            logger.info("="*50)
            if not download_sample_datasets():
                logger.error("Data preparation failed")
                success = False

        if success and args.step in ["all", "test"] and not args.skip_test:
            logger.info("\n" + "="*50)
            logger.info("STEP 3: Running Quick Test")
            logger.info("="*50)
            if not run_quick_test():
                logger.error("Quick test failed")
                success = False

        if success and args.step in ["all", "quick", "main", "full"]:
            logger.info("\n" + "="*50)
            logger.info(f"STEP 4: Running {args.step.upper()} Experiments")
            logger.info("="*50)

            experiment_type = args.step if args.step in ["quick", "main", "full"] else "main"
            if not run_experiments(experiment_type):
                logger.error("Experiments failed")
                success = False

        if success and args.step in ["all", "viz"]:
            logger.info("\n" + "="*50)
            logger.info("STEP 5: Generating Visualizations")
            logger.info("="*50)
            if not generate_visualizations():
                logger.error("Visualization generation failed")
                success = False

        # Final summary
        logger.info("\n" + "="*80)
        if success:
            logger.info("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
            logger.info("="*80)

            # Print summary of outputs
            logger.info("\nGenerated Outputs:")

            # Check for results
            results_dir = Path("results")
            if results_dir.exists():
                logger.info(f"📊 Results directory: {results_dir.absolute()}")

                # List result files
                for subdir in ["comprehensive", "figures", "logs"]:
                    subdir_path = results_dir / subdir
                    if subdir_path.exists():
                        files = list(subdir_path.glob("*"))
                        if files:
                            logger.info(f"   {subdir}/: {len(files)} files")
                            for file in files[:3]:  # Show first 3 files
                                logger.info(f"     - {file.name}")
                            if len(files) > 3:
                                logger.info(f"     ... and {len(files)-3} more")

            # Print next steps
            logger.info("\nNext Steps:")
            logger.info("1. Review results in results/comprehensive/")
            logger.info("2. Check figures in results/figures/")
            logger.info("3. Use results for your TNNLS paper")
            logger.info("4. Run additional experiments if needed")

        else:
            logger.error("❌ SOME STEPS FAILED!")
            logger.error("="*80)
            logger.error("Check the log file for detailed error information")
            logger.error(f"Log file: {log_file}")

        logger.info("="*80)

    except KeyboardInterrupt:
        logger.info("\n\nExperiment runner interrupted by user")
        success = False
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}", exc_info=True)
        success = False

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
