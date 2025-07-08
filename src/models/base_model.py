"""Base model class for forecasting models."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for forecasting models."""
    
    def __init__(self, random_state: Optional[int] = None):
        """Initialize base forecaster.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.is_fitted = False
        self.feature_names_ = None
        self.n_features_ = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseForecaster':
        """Fit the forecasting model to training data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make point predictions.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        pass
    
    def predict_intervals(self, X: np.ndarray, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            confidence_level: Confidence level for intervals (0 < confidence_level < 1)
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        predictions, uncertainties = self.predict_with_uncertainty(X)
        
        # Calculate confidence interval bounds
        alpha = 1 - confidence_level
        z_score = self._get_z_score(alpha / 2)
        
        lower_bounds = predictions - z_score * uncertainties
        upper_bounds = predictions + z_score * uncertainties
        
        return predictions, lower_bounds, upper_bounds
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the model and make predictions on the same data.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Predictions
        """
        self.fit(X, y)
        return self.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray, metric: str = 'rmse') -> float:
        """Score the model using specified metric.
        
        Args:
            X: Feature matrix
            y: True target values
            metric: Scoring metric ('rmse', 'mae', 'mape', 'r2')
            
        Returns:
            Score value
        """
        predictions = self.predict(X)
        return self._compute_metric(y, predictions, metric)
    
    def validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Validate input data.
        
        Args:
            X: Feature matrix
            y: Optional target values
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        
        if len(X) == 0:
            raise ValueError("X cannot be empty")
        
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values")
        
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values")
        
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise ValueError("y must be a numpy array")
            
            if y.ndim != 1:
                raise ValueError(f"y must be 1D array, got {y.ndim}D")
            
            if len(y) != len(X):
                raise ValueError(f"X and y must have same number of samples: {len(X)} vs {len(y)}")
            
            if np.any(np.isnan(y)):
                raise ValueError("y contains NaN values")
            
            if np.any(np.isinf(y)):
                raise ValueError("y contains infinite values")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {'random_state': self.random_state}
    
    def set_params(self, **params) -> 'BaseForecaster':
        """Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores (if supported by the model).
        
        Returns:
            Feature importance array or None if not supported
        """
        return None
    
    def _get_z_score(self, alpha: float) -> float:
        """Get z-score for given alpha level.
        
        Args:
            alpha: Alpha level (e.g., 0.025 for 95% confidence)
            
        Returns:
            Z-score value
        """
        from scipy import stats
        return stats.norm.ppf(1 - alpha)
    
    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        """Compute evaluation metric.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric: Metric name
            
        Returns:
            Metric value
        """
        if metric == 'rmse':
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif metric == 'mape':
            return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        elif metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-8))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def __repr__(self) -> str:
        """String representation of the model."""
        class_name = self.__class__.__name__
        params = self.get_params()
        param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        return f"{class_name}({param_str})"