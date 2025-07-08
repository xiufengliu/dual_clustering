"""Prediction interval evaluation utilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class IntervalEvaluator:
    """Specialized evaluator for prediction intervals."""
    
    def __init__(self):
        """Initialize interval evaluator."""
        pass
    
    def evaluate_intervals(self, y_true: np.ndarray, lower_bounds: np.ndarray,
                          upper_bounds: np.ndarray, confidence_level: float = 0.95) -> Dict[str, float]:
        """Evaluate prediction intervals comprehensively.
        
        Args:
            y_true: True values
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals
            confidence_level: Nominal confidence level
            
        Returns:
            Dictionary with interval evaluation metrics
        """
        metrics = {}
        
        # Basic interval metrics
        metrics.update(self._calculate_basic_metrics(y_true, lower_bounds, upper_bounds, confidence_level))
        
        # Coverage analysis
        metrics.update(self._analyze_coverage(y_true, lower_bounds, upper_bounds))
        
        # Width analysis
        metrics.update(self._analyze_width(lower_bounds, upper_bounds, y_true))
        
        # Calibration analysis
        metrics.update(self._analyze_calibration(y_true, lower_bounds, upper_bounds, confidence_level))
        
        return metrics
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, lower_bounds: np.ndarray,
                               upper_bounds: np.ndarray, confidence_level: float) -> Dict[str, float]:
        """Calculate basic interval metrics."""
        # Coverage indicators
        coverage = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        
        # Prediction Interval Coverage Probability (PICP)
        picp = np.mean(coverage)
        
        # Prediction Interval Normalized Average Width (PINAW)
        widths = upper_bounds - lower_bounds
        pinaw = np.mean(widths) / (np.max(y_true) - np.min(y_true))
        
        # Coverage Width-based Criterion (CWC)
        eta = 0.1  # Penalty parameter
        cwc = pinaw * (1 + eta * np.exp(-eta * (picp - confidence_level)))
        
        # Average Coverage Error (ACE)
        ace = abs(picp - confidence_level)
        
        # Interval Score (Winkler Score)
        alpha = 1 - confidence_level
        interval_score = self._calculate_interval_score(y_true, lower_bounds, upper_bounds, alpha)
        
        return {
            'picp': picp,
            'pinaw': pinaw,
            'cwc': cwc,
            'ace': ace,
            'interval_score': interval_score
        }
    
    def _analyze_coverage(self, y_true: np.ndarray, lower_bounds: np.ndarray,
                         upper_bounds: np.ndarray) -> Dict[str, float]:
        """Analyze coverage patterns."""
        coverage = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        
        # Coverage by quantiles
        n_quantiles = 5
        quantile_edges = np.linspace(0, 100, n_quantiles + 1)
        coverage_by_quantile = {}
        
        for i in range(n_quantiles):
            lower_q = np.percentile(y_true, quantile_edges[i])
            upper_q = np.percentile(y_true, quantile_edges[i + 1])
            
            mask = (y_true >= lower_q) & (y_true <= upper_q)
            if np.sum(mask) > 0:
                coverage_by_quantile[f'quantile_{i+1}'] = np.mean(coverage[mask])
        
        # Under-coverage (below lower bound)
        under_coverage = np.mean(y_true < lower_bounds)
        
        # Over-coverage (above upper bound)
        over_coverage = np.mean(y_true > upper_bounds)
        
        return {
            'under_coverage': under_coverage,
            'over_coverage': over_coverage,
            **coverage_by_quantile
        }
    
    def _analyze_width(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray,
                      y_true: np.ndarray) -> Dict[str, float]:
        """Analyze interval width patterns."""
        widths = upper_bounds - lower_bounds
        
        # Basic width statistics
        width_stats = {
            'mean_width': np.mean(widths),
            'std_width': np.std(widths),
            'min_width': np.min(widths),
            'max_width': np.max(widths),
            'median_width': np.median(widths)
        }
        
        # Width relative to target range
        target_range = np.max(y_true) - np.min(y_true)
        width_stats['relative_width'] = np.mean(widths) / target_range
        
        # Width variability
        width_stats['width_cv'] = np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 0
        
        return width_stats
    
    def _analyze_calibration(self, y_true: np.ndarray, lower_bounds: np.ndarray,
                           upper_bounds: np.ndarray, confidence_level: float) -> Dict[str, float]:
        """Analyze interval calibration."""
        # Conditional coverage analysis
        n_bins = 10
        calibration_metrics = {}
        
        # Sort by interval width
        widths = upper_bounds - lower_bounds
        width_order = np.argsort(widths)
        
        bin_size = len(y_true) // n_bins
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(y_true)
            
            bin_indices = width_order[start_idx:end_idx]
            
            if len(bin_indices) > 0:
                bin_coverage = np.mean(
                    (y_true[bin_indices] >= lower_bounds[bin_indices]) &
                    (y_true[bin_indices] <= upper_bounds[bin_indices])
                )
                calibration_metrics[f'coverage_bin_{i+1}'] = bin_coverage
        
        # Reliability (how close coverage is to nominal across bins)
        bin_coverages = [v for k, v in calibration_metrics.items() if k.startswith('coverage_bin_')]
        if bin_coverages:
            reliability = np.mean([(cov - confidence_level)**2 for cov in bin_coverages])
            calibration_metrics['reliability'] = reliability
        
        return calibration_metrics
    
    def _calculate_interval_score(self, y_true: np.ndarray, lower_bounds: np.ndarray,
                                upper_bounds: np.ndarray, alpha: float) -> float:
        """Calculate interval score (Winkler score)."""
        widths = upper_bounds - lower_bounds
        
        # Penalties for under-coverage
        under_penalty = 2 * alpha * np.maximum(0, lower_bounds - y_true)
        over_penalty = 2 * alpha * np.maximum(0, y_true - upper_bounds)
        
        # Total score
        scores = widths + under_penalty + over_penalty
        
        return np.mean(scores)
    
    def compare_interval_methods(self, y_true: np.ndarray,
                               interval_predictions: Dict[str, Dict[str, np.ndarray]],
                               confidence_level: float = 0.95) -> Dict[str, Any]:
        """Compare multiple interval prediction methods.
        
        Args:
            y_true: True values
            interval_predictions: Dict of method_name -> {'lower_bounds', 'upper_bounds'}
            confidence_level: Nominal confidence level
            
        Returns:
            Comparison results
        """
        results = {
            'individual_results': {},
            'rankings': {},
            'summary': {}
        }
        
        # Evaluate each method
        for method_name, intervals in interval_predictions.items():
            lower_bounds = intervals['lower_bounds']
            upper_bounds = intervals['upper_bounds']
            
            results['individual_results'][method_name] = self.evaluate_intervals(
                y_true, lower_bounds, upper_bounds, confidence_level
            )
        
        # Create rankings
        metrics_to_rank = ['picp', 'pinaw', 'cwc', 'ace', 'interval_score']
        
        for metric in metrics_to_rank:
            method_scores = []
            
            for method_name, method_results in results['individual_results'].items():
                if metric in method_results:
                    score = method_results[metric]
                    method_scores.append({'method': method_name, 'score': score})
            
            if method_scores:
                # Sort by score (descending for PICP, ascending for others)
                reverse = (metric == 'picp')
                sorted_scores = sorted(method_scores, key=lambda x: x['score'], reverse=reverse)
                results['rankings'][metric] = sorted_scores
        
        # Summary statistics
        if results['individual_results']:
            all_picps = [res['picp'] for res in results['individual_results'].values()]
            all_pinaws = [res['pinaw'] for res in results['individual_results'].values()]
            
            results['summary'] = {
                'best_coverage_method': max(results['individual_results'].items(), 
                                          key=lambda x: x[1]['picp'])[0],
                'narrowest_intervals_method': min(results['individual_results'].items(),
                                                key=lambda x: x[1]['pinaw'])[0],
                'coverage_range': (min(all_picps), max(all_picps)),
                'width_range': (min(all_pinaws), max(all_pinaws))
            }
        
        return results
    
    def generate_interval_report(self, evaluation_results: Dict[str, Any],
                               confidence_level: float = 0.95) -> str:
        """Generate a text report for interval evaluation.
        
        Args:
            evaluation_results: Results from evaluate_intervals or compare_interval_methods
            confidence_level: Nominal confidence level
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*60)
        report.append("PREDICTION INTERVAL EVALUATION REPORT")
        report.append("="*60)
        report.append(f"Nominal Confidence Level: {confidence_level:.1%}")
        
        if 'individual_results' in evaluation_results:
            # Multiple methods comparison
            report.append("\nMETHOD COMPARISON:")
            report.append("-" * 30)
            
            for method_name, results in evaluation_results['individual_results'].items():
                report.append(f"\n{method_name}:")
                report.append(f"  Coverage (PICP): {results['picp']:.3f}")
                report.append(f"  Width (PINAW):   {results['pinaw']:.3f}")
                report.append(f"  CWC Score:       {results['cwc']:.3f}")
                report.append(f"  Interval Score:  {results['interval_score']:.3f}")
                
                # Coverage analysis
                if 'under_coverage' in results:
                    report.append(f"  Under-coverage:  {results['under_coverage']:.3f}")
                    report.append(f"  Over-coverage:   {results['over_coverage']:.3f}")
            
            # Rankings
            if 'rankings' in evaluation_results:
                report.append("\nRANKINGS:")
                report.append("-" * 20)
                
                for metric, ranking in evaluation_results['rankings'].items():
                    report.append(f"\n{metric.upper()}:")
                    for i, method_info in enumerate(ranking, 1):
                        report.append(f"  {i}. {method_info['method']}: {method_info['score']:.4f}")
        
        else:
            # Single method evaluation
            report.append("\nEVALUATION RESULTS:")
            report.append("-" * 30)
            
            for metric, value in evaluation_results.items():
                if isinstance(value, (int, float)):
                    report.append(f"{metric.upper()}: {value:.4f}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
