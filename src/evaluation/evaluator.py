"""Forecast evaluation utilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from .metrics import ForecastingMetrics
from .statistical_tests import StatisticalTests

logger = logging.getLogger(__name__)


class ForecastEvaluator:
    """Comprehensive forecast evaluation framework."""
    
    def __init__(self, alpha: float = 0.05):
        """Initialize forecast evaluator.
        
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
        self.metrics_calculator = ForecastingMetrics()
        self.statistical_tests = StatisticalTests(alpha=alpha)
        
    def evaluate_single_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                             lower_bounds: Optional[np.ndarray] = None,
                             upper_bounds: Optional[np.ndarray] = None,
                             confidence_level: float = 0.95) -> Dict[str, Any]:
        """Evaluate a single forecasting model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        # Point forecast metrics
        results['point_metrics'] = self.metrics_calculator.calculate_point_metrics(y_true, y_pred)
        
        # Interval metrics if bounds provided
        if lower_bounds is not None and upper_bounds is not None:
            results['interval_metrics'] = self.metrics_calculator.calculate_interval_metrics(
                y_true, lower_bounds, upper_bounds, confidence_level
            )
        
        # Residual analysis
        residuals = y_true - y_pred
        results['residual_analysis'] = self._analyze_residuals(residuals)
        
        # Error distribution
        results['error_distribution'] = self._analyze_error_distribution(residuals)
        
        return results
    
    def compare_models(self, y_true: np.ndarray, 
                      predictions: Dict[str, np.ndarray],
                      reference_model: Optional[str] = None) -> Dict[str, Any]:
        """Compare multiple forecasting models.
        
        Args:
            y_true: True values
            predictions: Dictionary of model_name -> predictions
            reference_model: Reference model for pairwise comparisons
            
        Returns:
            Dictionary with comparison results
        """
        results = {
            'individual_results': {},
            'rankings': {},
            'statistical_tests': {}
        }
        
        # Evaluate each model individually
        for model_name, y_pred in predictions.items():
            results['individual_results'][model_name] = self.evaluate_single_model(y_true, y_pred)
        
        # Create rankings
        results['rankings'] = self._create_rankings(results['individual_results'])
        
        # Statistical tests
        errors_dict = {name: y_true - pred for name, pred in predictions.items()}
        
        if len(errors_dict) >= 2:
            # Comprehensive model comparison
            if reference_model and reference_model in errors_dict:
                comparison_results = self.statistical_tests.comprehensive_model_comparison(
                    errors_dict, reference_model=reference_model
                )
                results['statistical_tests'] = comparison_results
            else:
                # Use best model as reference
                best_model = results['rankings']['rmse'][0]['model']
                comparison_results = self.statistical_tests.comprehensive_model_comparison(
                    errors_dict, reference_model=best_model
                )
                results['statistical_tests'] = comparison_results
        
        return results
    
    def evaluate_with_intervals(self, y_true: np.ndarray,
                               predictions: Dict[str, Dict[str, np.ndarray]],
                               confidence_level: float = 0.95) -> Dict[str, Any]:
        """Evaluate models with prediction intervals.
        
        Args:
            y_true: True values
            predictions: Dictionary of model_name -> {'predictions', 'lower_bounds', 'upper_bounds'}
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with evaluation results including interval metrics
        """
        results = {
            'individual_results': {},
            'rankings': {},
            'interval_rankings': {}
        }
        
        # Evaluate each model
        for model_name, pred_dict in predictions.items():
            y_pred = pred_dict['predictions']
            lower_bounds = pred_dict.get('lower_bounds')
            upper_bounds = pred_dict.get('upper_bounds')
            
            results['individual_results'][model_name] = self.evaluate_single_model(
                y_true, y_pred, lower_bounds, upper_bounds, confidence_level
            )
        
        # Create rankings for point metrics
        results['rankings'] = self._create_rankings(results['individual_results'])
        
        # Create rankings for interval metrics
        results['interval_rankings'] = self._create_interval_rankings(results['individual_results'])
        
        return results
    
    def _analyze_residuals(self, residuals: np.ndarray) -> Dict[str, float]:
        """Analyze residual properties."""
        analysis = {}
        
        # Basic statistics
        analysis['mean'] = np.mean(residuals)
        analysis['std'] = np.std(residuals)
        analysis['skewness'] = self._calculate_skewness(residuals)
        analysis['kurtosis'] = self._calculate_kurtosis(residuals)
        
        # Autocorrelation (lag-1)
        if len(residuals) > 1:
            analysis['autocorr_lag1'] = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        else:
            analysis['autocorr_lag1'] = 0.0
        
        # Ljung-Box test for autocorrelation (if statsmodels available)
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
            analysis['ljung_box_pvalue'] = lb_result['lb_pvalue'].iloc[-1]
        except ImportError:
            analysis['ljung_box_pvalue'] = None
        
        return analysis
    
    def _analyze_error_distribution(self, errors: np.ndarray) -> Dict[str, Any]:
        """Analyze error distribution."""
        abs_errors = np.abs(errors)
        
        distribution = {
            'percentiles': {
                '5th': np.percentile(abs_errors, 5),
                '25th': np.percentile(abs_errors, 25),
                '50th': np.percentile(abs_errors, 50),
                '75th': np.percentile(abs_errors, 75),
                '95th': np.percentile(abs_errors, 95)
            },
            'max_error': np.max(abs_errors),
            'min_error': np.min(abs_errors),
            'error_variance': np.var(errors)
        }
        
        return distribution
    
    def _create_rankings(self, individual_results: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """Create model rankings based on different metrics."""
        rankings = {}
        
        metrics_to_rank = ['rmse', 'mae', 'mape', 'r2']
        
        for metric in metrics_to_rank:
            model_scores = []
            
            for model_name, results in individual_results.items():
                if 'point_metrics' in results and metric in results['point_metrics']:
                    score = results['point_metrics'][metric]
                    model_scores.append({'model': model_name, 'score': score})
            
            if model_scores:
                # Sort by score (ascending for error metrics, descending for R²)
                reverse = (metric == 'r2')
                sorted_scores = sorted(model_scores, key=lambda x: x['score'], reverse=reverse)
                rankings[metric] = sorted_scores
        
        return rankings
    
    def _create_interval_rankings(self, individual_results: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """Create model rankings based on interval metrics."""
        rankings = {}
        
        interval_metrics = ['picp', 'pinaw', 'cwc', 'interval_score']
        
        for metric in interval_metrics:
            model_scores = []
            
            for model_name, results in individual_results.items():
                if ('interval_metrics' in results and 
                    results['interval_metrics'] and 
                    metric in results['interval_metrics']):
                    score = results['interval_metrics'][metric]
                    model_scores.append({'model': model_name, 'score': score})
            
            if model_scores:
                # Sort by score (ascending for most metrics, descending for PICP)
                reverse = (metric == 'picp')
                sorted_scores = sorted(model_scores, key=lambda x: x['score'], reverse=reverse)
                rankings[metric] = sorted_scores
        
        return rankings
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        return kurtosis
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any],
                                 model_names: Optional[List[str]] = None) -> str:
        """Generate a text report of evaluation results.
        
        Args:
            evaluation_results: Results from compare_models or evaluate_with_intervals
            model_names: Optional list of model names to include
            
        Returns:
            Formatted evaluation report
        """
        report = []
        report.append("="*80)
        report.append("FORECASTING MODEL EVALUATION REPORT")
        report.append("="*80)
        
        # Model rankings
        if 'rankings' in evaluation_results:
            report.append("\nMODEL RANKINGS:")
            report.append("-" * 40)
            
            for metric, ranking in evaluation_results['rankings'].items():
                report.append(f"\n{metric.upper()} Rankings:")
                for i, model_info in enumerate(ranking[:5], 1):
                    report.append(f"  {i}. {model_info['model']}: {model_info['score']:.4f}")
        
        # Statistical significance
        if 'statistical_tests' in evaluation_results:
            report.append("\nSTATISTICAL SIGNIFICANCE TESTS:")
            report.append("-" * 40)
            
            if 'pairwise_comparisons' in evaluation_results['statistical_tests']:
                report.append("\nPairwise Comparisons (Diebold-Mariano Test):")
                for comparison, result in evaluation_results['statistical_tests']['pairwise_comparisons'].items():
                    significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
                    report.append(f"  {comparison}: p={result['p_value']:.4f} {significance}")
        
        # Individual model results
        if 'individual_results' in evaluation_results:
            report.append("\nINDIVIDUAL MODEL RESULTS:")
            report.append("-" * 40)
            
            for model_name, results in evaluation_results['individual_results'].items():
                if model_names is None or model_name in model_names:
                    report.append(f"\n{model_name}:")
                    
                    if 'point_metrics' in results:
                        metrics = results['point_metrics']
                        report.append(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                        report.append(f"  MAE:  {metrics.get('mae', 'N/A'):.4f}")
                        report.append(f"  MAPE: {metrics.get('mape', 'N/A'):.2f}%")
                        report.append(f"  R²:   {metrics.get('r2', 'N/A'):.4f}")
                    
                    if 'interval_metrics' in results and results['interval_metrics']:
                        interval_metrics = results['interval_metrics']
                        report.append(f"  PICP: {interval_metrics.get('picp', 'N/A'):.3f}")
                        report.append(f"  PINAW: {interval_metrics.get('pinaw', 'N/A'):.3f}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
