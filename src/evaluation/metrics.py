"""Evaluation metrics for renewable energy forecasting."""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ForecastingMetrics:
    """Comprehensive metrics for evaluating forecasting performance."""
    
    def __init__(self):
        """Initialize forecasting metrics calculator."""
        pass
    
    def calculate_point_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate point forecasting metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with point forecasting metrics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        metrics = {}
        
        # Root Mean Squared Error
        metrics['rmse'] = self.rmse(y_true, y_pred)
        
        # Mean Absolute Error
        metrics['mae'] = self.mae(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        metrics['mape'] = self.mape(y_true, y_pred)
        
        # Symmetric Mean Absolute Percentage Error
        metrics['smape'] = self.smape(y_true, y_pred)
        
        # R-squared
        metrics['r2'] = self.r2_score(y_true, y_pred)
        
        # Mean Bias Error
        metrics['mbe'] = self.mbe(y_true, y_pred)
        
        # Normalized RMSE
        metrics['nrmse'] = self.nrmse(y_true, y_pred)
        
        # Coefficient of Variation of RMSE
        metrics['cv_rmse'] = self.cv_rmse(y_true, y_pred)
        
        return metrics
    
    def calculate_interval_metrics(self, y_true: np.ndarray, 
                                 lower_bounds: np.ndarray, 
                                 upper_bounds: np.ndarray,
                                 confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate prediction interval metrics.
        
        Args:
            y_true: True values
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals
            confidence_level: Nominal confidence level
            
        Returns:
            Dictionary with interval metrics
        """
        metrics = {}
        
        # Prediction Interval Coverage Probability
        metrics['picp'] = self.picp(y_true, lower_bounds, upper_bounds)
        
        # Prediction Interval Normalized Average Width
        metrics['pinaw'] = self.pinaw(y_true, lower_bounds, upper_bounds)
        
        # Coverage Width-based Criterion
        metrics['cwc'] = self.cwc(y_true, lower_bounds, upper_bounds, confidence_level)
        
        # Average Coverage Error
        metrics['ace'] = self.ace(y_true, lower_bounds, upper_bounds, confidence_level)
        
        # Interval Score
        metrics['interval_score'] = self.interval_score(y_true, lower_bounds, upper_bounds, confidence_level)
        
        return metrics
    
    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
        
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    def smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        
        if not np.any(mask):
            return 0.0
        
        return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)
    
    def r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared coefficient of determination."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return float(1 - (ss_res / ss_tot))
    
    def mbe(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Bias Error."""
        return float(np.mean(y_pred - y_true))
    
    def nrmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Normalized Root Mean Squared Error."""
        rmse_val = self.rmse(y_true, y_pred)
        y_range = np.max(y_true) - np.min(y_true)
        
        if y_range == 0:
            return 0.0 if rmse_val == 0 else float('inf')
        
        return float(rmse_val / y_range)
    
    def cv_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Coefficient of Variation of RMSE."""
        rmse_val = self.rmse(y_true, y_pred)
        mean_true = np.mean(y_true)
        
        if mean_true == 0:
            return float('inf') if rmse_val > 0 else 0.0
        
        return float(rmse_val / mean_true * 100)
    
    def picp(self, y_true: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> float:
        """Prediction Interval Coverage Probability."""
        coverage = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        return float(np.mean(coverage))
    
    def pinaw(self, y_true: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> float:
        """Prediction Interval Normalized Average Width."""
        interval_widths = upper_bounds - lower_bounds
        y_range = np.max(y_true) - np.min(y_true)
        
        if y_range == 0:
            return 0.0 if np.all(interval_widths == 0) else float('inf')
        
        return float(np.mean(interval_widths) / y_range)
    
    def cwc(self, y_true: np.ndarray, lower_bounds: np.ndarray, 
            upper_bounds: np.ndarray, confidence_level: float) -> float:
        """Coverage Width-based Criterion."""
        picp_val = self.picp(y_true, lower_bounds, upper_bounds)
        pinaw_val = self.pinaw(y_true, lower_bounds, upper_bounds)
        
        # Penalty for coverage below nominal level
        eta = 50  # Penalty parameter
        coverage_penalty = 0
        
        if picp_val < confidence_level:
            coverage_penalty = eta * (confidence_level - picp_val)
        
        return float(pinaw_val + coverage_penalty)
    
    def ace(self, y_true: np.ndarray, lower_bounds: np.ndarray, 
            upper_bounds: np.ndarray, confidence_level: float) -> float:
        """Average Coverage Error."""
        picp_val = self.picp(y_true, lower_bounds, upper_bounds)
        return float(abs(picp_val - confidence_level))
    
    def interval_score(self, y_true: np.ndarray, lower_bounds: np.ndarray, 
                      upper_bounds: np.ndarray, confidence_level: float) -> float:
        """Interval Score (lower is better)."""
        alpha = 1 - confidence_level
        interval_widths = upper_bounds - lower_bounds
        
        # Penalties for being outside the interval
        lower_penalty = 2 * alpha * np.maximum(0, lower_bounds - y_true)
        upper_penalty = 2 * alpha * np.maximum(0, y_true - upper_bounds)
        
        scores = interval_widths + lower_penalty + upper_penalty
        return float(np.mean(scores))
    
    def calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (percentage of correct direction predictions)."""
        if len(y_true) < 2:
            return float('nan')
        
        true_directions = np.diff(y_true) > 0
        pred_directions = np.diff(y_pred) > 0
        
        correct_directions = true_directions == pred_directions
        return float(np.mean(correct_directions))
    
    def calculate_quantile_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]) -> Dict[str, float]:
        """Calculate metrics at different quantiles of the error distribution."""
        errors = np.abs(y_true - y_pred)
        
        quantile_metrics = {}
        for q in quantiles:
            quantile_metrics[f'error_q{int(q*100)}'] = float(np.percentile(errors, q * 100))
        
        return quantile_metrics
    
    def calculate_seasonal_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 season_indices: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for different seasons/periods."""
        if season_indices is None:
            # Simple quarterly split
            n_samples = len(y_true)
            season_indices = np.repeat(np.arange(4), n_samples // 4 + 1)[:n_samples]
        
        seasonal_metrics = {}
        
        for season in np.unique(season_indices):
            mask = season_indices == season
            if np.sum(mask) > 0:
                seasonal_metrics[f'season_{season}'] = self.calculate_point_metrics(
                    y_true[mask], y_pred[mask]
                )
        
        return seasonal_metrics
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      lower_bounds: Optional[np.ndarray] = None,
                                      upper_bounds: Optional[np.ndarray] = None,
                                      confidence_level: float = 0.95) -> Dict[str, any]:
        """Calculate comprehensive set of metrics."""
        metrics = {}
        
        # Point metrics
        metrics['point_metrics'] = self.calculate_point_metrics(y_true, y_pred)
        
        # Interval metrics (if bounds provided)
        if lower_bounds is not None and upper_bounds is not None:
            metrics['interval_metrics'] = self.calculate_interval_metrics(
                y_true, lower_bounds, upper_bounds, confidence_level
            )
        
        # Directional accuracy
        metrics['directional_accuracy'] = self.calculate_directional_accuracy(y_true, y_pred)
        
        # Quantile metrics
        metrics['quantile_metrics'] = self.calculate_quantile_metrics(y_true, y_pred)
        
        # Summary statistics
        metrics['summary'] = {
            'n_samples': len(y_true),
            'y_true_mean': float(np.mean(y_true)),
            'y_true_std': float(np.std(y_true)),
            'y_pred_mean': float(np.mean(y_pred)),
            'y_pred_std': float(np.std(y_pred)),
            'correlation': float(np.corrcoef(y_true, y_pred)[0, 1])
        }
        
        return metrics