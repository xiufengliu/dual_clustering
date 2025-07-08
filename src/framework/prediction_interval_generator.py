"""Prediction interval generation for forecasting models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator
import logging

logger = logging.getLogger(__name__)


class PredictionIntervalGenerator:
    """Generate prediction intervals for forecasting models."""
    
    def __init__(self, method: str = 'heuristic', confidence_level: float = 0.95):
        """Initialize prediction interval generator.
        
        Args:
            method: Method for generating intervals ('heuristic', 'bootstrap', 'quantile')
            confidence_level: Confidence level for intervals
        """
        self.method = method
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        # Parameters for different methods
        self.gamma = 1.96  # Z-score for normal approximation
        self.beta = 1.0    # Scaling factor
        
        # Fitted parameters
        self.residual_std = None
        self.quantile_models = {}
        self.bootstrap_samples = None
        
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray, 
            X: Optional[np.ndarray] = None, model: Optional[BaseEstimator] = None):
        """Fit the interval generator on training data.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            X: Features (for quantile regression)
            model: Fitted model (for bootstrap)
        """
        residuals = y_true - y_pred
        self.residual_std = np.std(residuals)
        
        if self.method == 'quantile' and X is not None:
            self._fit_quantile_models(X, residuals)
        elif self.method == 'bootstrap' and model is not None and X is not None:
            self._fit_bootstrap(X, y_true, model)
        
        logger.info(f"Fitted {self.method} interval generator with std={self.residual_std:.4f}")
    
    def generate_intervals(self, predictions: np.ndarray, 
                          uncertainties: Optional[np.ndarray] = None,
                          X: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals.
        
        Args:
            predictions: Point predictions
            uncertainties: Uncertainty estimates (if available)
            X: Features (for quantile regression)
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if self.method == 'heuristic':
            return self._generate_heuristic_intervals(predictions, uncertainties)
        elif self.method == 'quantile':
            return self._generate_quantile_intervals(predictions, X)
        elif self.method == 'bootstrap':
            return self._generate_bootstrap_intervals(predictions, X)
        else:
            raise ValueError(f"Unknown interval generation method: {self.method}")
    
    def _generate_heuristic_intervals(self, predictions: np.ndarray,
                                    uncertainties: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate intervals using heuristic method."""
        if uncertainties is not None:
            # Use provided uncertainties
            interval_width = self.gamma * uncertainties
        else:
            # Use residual standard deviation
            interval_width = self.gamma * self.residual_std
        
        # Apply scaling factor
        interval_width *= self.beta
        
        lower_bounds = predictions - interval_width
        upper_bounds = predictions + interval_width
        
        return lower_bounds, upper_bounds
    
    def _generate_quantile_intervals(self, predictions: np.ndarray,
                                   X: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate intervals using quantile regression."""
        if not self.quantile_models or X is None:
            # Fall back to heuristic method
            logger.warning("Quantile models not fitted or X not provided, falling back to heuristic")
            return self._generate_heuristic_intervals(predictions)
        
        try:
            lower_quantile = self.alpha / 2
            upper_quantile = 1 - self.alpha / 2
            
            lower_residuals = self.quantile_models[lower_quantile].predict(X)
            upper_residuals = self.quantile_models[upper_quantile].predict(X)
            
            lower_bounds = predictions + lower_residuals
            upper_bounds = predictions + upper_residuals
            
            return lower_bounds, upper_bounds
            
        except Exception as e:
            logger.error(f"Quantile interval generation failed: {e}")
            return self._generate_heuristic_intervals(predictions)
    
    def _generate_bootstrap_intervals(self, predictions: np.ndarray,
                                    X: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate intervals using bootstrap method."""
        if self.bootstrap_samples is None or X is None:
            logger.warning("Bootstrap samples not available or X not provided, falling back to heuristic")
            return self._generate_heuristic_intervals(predictions)
        
        try:
            # Use bootstrap samples to estimate prediction distribution
            n_bootstrap = len(self.bootstrap_samples)
            bootstrap_predictions = np.zeros((n_bootstrap, len(predictions)))
            
            for i, sample_indices in enumerate(self.bootstrap_samples):
                # This is a simplified bootstrap - in practice, you'd retrain models
                # For now, add noise based on bootstrap sample variance
                noise = np.random.normal(0, self.residual_std, len(predictions))
                bootstrap_predictions[i] = predictions + noise
            
            # Calculate quantiles
            lower_quantile = self.alpha / 2
            upper_quantile = 1 - self.alpha / 2
            
            lower_bounds = np.percentile(bootstrap_predictions, lower_quantile * 100, axis=0)
            upper_bounds = np.percentile(bootstrap_predictions, upper_quantile * 100, axis=0)
            
            return lower_bounds, upper_bounds
            
        except Exception as e:
            logger.error(f"Bootstrap interval generation failed: {e}")
            return self._generate_heuristic_intervals(predictions)
    
    def _fit_quantile_models(self, X: np.ndarray, residuals: np.ndarray):
        """Fit quantile regression models for residuals."""
        try:
            from sklearn.linear_model import QuantileRegressor
            
            quantiles = [self.alpha / 2, 1 - self.alpha / 2]
            
            for quantile in quantiles:
                model = QuantileRegressor(quantile=quantile, alpha=0.01)
                model.fit(X, residuals)
                self.quantile_models[quantile] = model
                
            logger.info(f"Fitted quantile models for quantiles: {quantiles}")
            
        except ImportError:
            logger.warning("QuantileRegressor not available, falling back to heuristic method")
            self.method = 'heuristic'
        except Exception as e:
            logger.error(f"Failed to fit quantile models: {e}")
            self.method = 'heuristic'
    
    def _fit_bootstrap(self, X: np.ndarray, y: np.ndarray, model: BaseEstimator):
        """Fit bootstrap samples."""
        try:
            n_bootstrap = 100
            n_samples = len(X)
            
            self.bootstrap_samples = []
            
            for _ in range(n_bootstrap):
                # Generate bootstrap sample indices
                sample_indices = np.random.choice(n_samples, n_samples, replace=True)
                self.bootstrap_samples.append(sample_indices)
            
            logger.info(f"Generated {n_bootstrap} bootstrap samples")
            
        except Exception as e:
            logger.error(f"Failed to generate bootstrap samples: {e}")
            self.method = 'heuristic'
    
    def set_parameters(self, gamma: Optional[float] = None, beta: Optional[float] = None):
        """Set interval generation parameters.
        
        Args:
            gamma: Z-score multiplier
            beta: Scaling factor
        """
        if gamma is not None:
            self.gamma = gamma
        if beta is not None:
            self.beta = beta
        
        logger.info(f"Updated parameters: gamma={self.gamma}, beta={self.beta}")
    
    def adaptive_intervals(self, predictions: np.ndarray, 
                          indeterminacy: np.ndarray,
                          base_width: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate adaptive intervals based on indeterminacy.
        
        Args:
            predictions: Point predictions
            indeterminacy: Indeterminacy values from neutrosophic transformation
            base_width: Base interval width (if None, uses residual_std)
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if base_width is None:
            base_width = self.residual_std if self.residual_std is not None else np.std(predictions) * 0.1
        
        # Scale interval width based on indeterminacy
        # Higher indeterminacy -> wider intervals
        adaptive_width = base_width * (1 + self.beta * indeterminacy)
        
        # Apply confidence level scaling
        interval_width = self.gamma * adaptive_width
        
        lower_bounds = predictions - interval_width
        upper_bounds = predictions + interval_width
        
        return lower_bounds, upper_bounds
    
    def evaluate_intervals(self, y_true: np.ndarray, lower_bounds: np.ndarray,
                          upper_bounds: np.ndarray) -> Dict[str, float]:
        """Evaluate interval quality.
        
        Args:
            y_true: True values
            lower_bounds: Lower bounds
            upper_bounds: Upper bounds
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Coverage
        coverage = np.mean((y_true >= lower_bounds) & (y_true <= upper_bounds))
        
        # Average width
        avg_width = np.mean(upper_bounds - lower_bounds)
        
        # Normalized width
        y_range = np.max(y_true) - np.min(y_true)
        normalized_width = avg_width / y_range if y_range > 0 else 0
        
        # Coverage Width Criterion
        eta = 0.1
        cwc = normalized_width * (1 + eta * np.exp(-eta * (coverage - self.confidence_level)))
        
        return {
            'coverage': coverage,
            'average_width': avg_width,
            'normalized_width': normalized_width,
            'cwc': cwc,
            'target_coverage': self.confidence_level
        }
    
    def calibrate_parameters(self, y_true: np.ndarray, predictions: np.ndarray,
                           uncertainties: Optional[np.ndarray] = None,
                           target_coverage: Optional[float] = None) -> Dict[str, float]:
        """Calibrate interval parameters to achieve target coverage.
        
        Args:
            y_true: True values
            predictions: Predictions
            uncertainties: Uncertainty estimates
            target_coverage: Target coverage (if None, uses confidence_level)
            
        Returns:
            Dictionary with calibrated parameters
        """
        if target_coverage is None:
            target_coverage = self.confidence_level
        
        # Grid search for optimal gamma
        gamma_values = np.linspace(0.5, 3.0, 20)
        best_gamma = self.gamma
        best_coverage_error = float('inf')
        
        for gamma in gamma_values:
            # Generate intervals with this gamma
            old_gamma = self.gamma
            self.gamma = gamma
            
            lower_bounds, upper_bounds = self._generate_heuristic_intervals(predictions, uncertainties)
            
            # Evaluate coverage
            coverage = np.mean((y_true >= lower_bounds) & (y_true <= upper_bounds))
            coverage_error = abs(coverage - target_coverage)
            
            if coverage_error < best_coverage_error:
                best_coverage_error = coverage_error
                best_gamma = gamma
            
            # Restore original gamma
            self.gamma = old_gamma
        
        # Update with best parameters
        self.gamma = best_gamma
        
        logger.info(f"Calibrated gamma to {best_gamma:.3f} for target coverage {target_coverage:.3f}")
        
        return {
            'gamma': best_gamma,
            'target_coverage': target_coverage,
            'achieved_coverage_error': best_coverage_error
        }
