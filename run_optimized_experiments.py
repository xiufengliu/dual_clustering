#!/usr/bin/env python3
"""
Optimized experiment runner with performance improvements.
This script runs experiments with reduced computational complexity for faster execution.
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from experiments.comprehensive_evaluation import ComprehensiveEvaluation
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

def main():
    """Main function to run optimized experiments."""
    parser = argparse.ArgumentParser(description="Run optimized experiments for performance testing")
    parser.add_argument("--config", type=str, default="fast_benchmark_config",
                       help="Configuration name (default: fast_benchmark_config)")
    parser.add_argument("--output-dir", type=str, default="results/optimized",
                       help="Output directory")
    parser.add_argument("--datasets", nargs="+",
                       default=[
                           'kaggle_solar_plant',  # Small dataset for testing
                           'entso_e_load_fixed',  # Small dataset
                           'nrel_canada_wind'     # Medium dataset
                       ],
                       help="Datasets to evaluate (default: small datasets for speed)")
    parser.add_argument("--max-samples", type=int, default=2000,
                       help="Maximum samples per dataset for speed testing")
    parser.add_argument("--skip-ablation", action="store_true",
                       help="Skip ablation studies")
    parser.add_argument("--skip-sensitivity", action="store_true",
                       help="Skip sensitivity analysis")
    parser.add_argument("--skip-computational", action="store_true",
                       help="Skip computational analysis")
    parser.add_argument("--skip-cross-dataset", action="store_true",
                       help="Skip cross-dataset evaluation")
    parser.add_argument("--skip-robustness", action="store_true",
                       help="Skip robustness analysis")
    parser.add_argument("--profile", action="store_true",
                       help="Enable performance profiling")

    args = parser.parse_args()

    # Setup logging
    log_file = Path(args.output_dir) / f"optimized_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger("optimized_experiments", log_file=log_file)

    logger.info("="*80)
    logger.info("OPTIMIZED EXPERIMENTS FOR PERFORMANCE TESTING")
    logger.info("="*80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Max samples per dataset: {args.max_samples}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config(experiment_name=args.config)
    
    # Override max samples if specified
    if args.max_samples:
        config['max_samples_per_dataset'] = args.max_samples
        logger.info(f"Overriding max samples per dataset to {args.max_samples}")

    # Initialize evaluation
    evaluation = ComprehensiveEvaluation(config, args.output_dir)

    # Performance tracking
    start_time = time.time()
    stage_times = {}

    try:
        # Run main experiments (always run this)
        logger.info("\n" + "="*50)
        logger.info("STAGE 1: Main Comparison Experiments")
        logger.info("="*50)
        stage_start = time.time()
        
        evaluation.results['main_results'] = evaluation.run_main_experiments(args.datasets)
        
        stage_times['main_experiments'] = time.time() - stage_start
        logger.info(f"Main experiments completed in {stage_times['main_experiments']:.2f} seconds")

        # Run ablation studies (optional)
        if not args.skip_ablation:
            logger.info("\n" + "="*50)
            logger.info("STAGE 2: Ablation Studies")
            logger.info("="*50)
            stage_start = time.time()

            try:
                # Use the first dataset for ablation studies
                evaluation.results['ablation_studies'] = evaluation.run_ablation_studies(args.datasets[0])
                logger.info(f"Ablation studies completed successfully")
            except Exception as e:
                logger.error(f"Ablation studies failed: {e}")
                evaluation.results['ablation_studies'] = {'error': str(e), 'failed': True}

            stage_times['ablation_studies'] = time.time() - stage_start
            logger.info(f"Ablation studies completed in {stage_times['ablation_studies']:.2f} seconds")

        # Run sensitivity analysis (optional)
        if not args.skip_sensitivity:
            logger.info("\n" + "="*50)
            logger.info("STAGE 3: Sensitivity Analysis")
            logger.info("="*50)
            stage_start = time.time()

            try:
                evaluation.results['sensitivity_analysis'] = evaluation.run_sensitivity_analysis(args.datasets[0])
                logger.info(f"Sensitivity analysis completed successfully")
            except Exception as e:
                logger.error(f"Sensitivity analysis failed: {e}")
                evaluation.results['sensitivity_analysis'] = {'error': str(e), 'failed': True}

            stage_times['sensitivity_analysis'] = time.time() - stage_start
            logger.info(f"Sensitivity analysis completed in {stage_times['sensitivity_analysis']:.2f} seconds")

        # Run computational analysis (optional)
        if not args.skip_computational:
            logger.info("\n" + "="*50)
            logger.info("STAGE 4: Computational Analysis")
            logger.info("="*50)
            stage_start = time.time()
            
            # Use smaller dataset sizes for computational analysis
            evaluation.results['computational_analysis'] = evaluation.run_computational_analysis([500, 1000, 2000])
            
            stage_times['computational_analysis'] = time.time() - stage_start
            logger.info(f"Computational analysis completed in {stage_times['computational_analysis']:.2f} seconds")

        # Run cross-dataset evaluation (optional)
        if not args.skip_cross_dataset and len(args.datasets) > 1:
            logger.info("\n" + "="*50)
            logger.info("STAGE 5: Cross-Dataset Evaluation")
            logger.info("="*50)
            stage_start = time.time()
            
            # Limit to first 2 datasets for speed
            evaluation.results['cross_dataset_results'] = evaluation.run_cross_dataset_evaluation(args.datasets[:2])
            
            stage_times['cross_dataset_evaluation'] = time.time() - stage_start
            logger.info(f"Cross-dataset evaluation completed in {stage_times['cross_dataset_evaluation']:.2f} seconds")

        # Run robustness analysis (optional)
        if not args.skip_robustness:
            logger.info("\n" + "="*50)
            logger.info("STAGE 6: Robustness Analysis")
            logger.info("="*50)
            stage_start = time.time()

            try:
                evaluation.results['robustness_analysis'] = evaluation.run_robustness_analysis(args.datasets[0])
                logger.info(f"Robustness analysis completed successfully")
            except Exception as e:
                logger.error(f"Robustness analysis failed: {e}")
                evaluation.results['robustness_analysis'] = {'error': str(e), 'failed': True}

            stage_times['robustness_analysis'] = time.time() - stage_start
            logger.info(f"Robustness analysis completed in {stage_times['robustness_analysis']:.2f} seconds")

        # Generate summary
        evaluation.results['summary'] = evaluation.generate_summary()

        # Add performance metrics to results
        total_time = time.time() - start_time
        evaluation.results['performance_metrics'] = {
            'total_execution_time': total_time,
            'stage_times': stage_times,
            'datasets_processed': len(args.datasets),
            'average_time_per_dataset': total_time / len(args.datasets) if args.datasets else 0
        }

        # Save results
        evaluation.save_results()

        # Print performance summary
        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*80)
        logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"Datasets processed: {len(args.datasets)}")
        logger.info(f"Average time per dataset: {total_time/len(args.datasets):.2f} seconds")
        
        logger.info("\nStage breakdown:")
        for stage, duration in stage_times.items():
            percentage = (duration / total_time) * 100
            logger.info(f"  {stage}: {duration:.2f}s ({percentage:.1f}%)")

        # Print results summary
        logger.info("\n" + "="*80)
        logger.info("RESULTS SUMMARY")
        logger.info("="*80)
        
        if 'main_results' in evaluation.results:
            logger.info("\nMain Results:")
            for dataset, results in evaluation.results['main_results'].items():
                if 'model_results' in results:
                    logger.info(f"\n{dataset.upper()}:")
                    for model_name, model_results in results['model_results'].items():
                        if 'point_metrics' in model_results and not model_results.get('failed', False):
                            rmse = model_results['point_metrics']['rmse']
                            logger.info(f"  {model_name}: RMSE = {rmse:.4f}")

        logger.info("\n" + "="*80)
        logger.info("🎉 OPTIMIZED EXPERIMENTS COMPLETED SUCCESSFULLY!")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Optimized experiments failed: {e}", exc_info=True)
        print(f"\nERROR: Optimized experiments failed: {e}")
        print("Check the log file for detailed error information.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
