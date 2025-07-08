"""Ensemble predictor for combining multiple forecasting models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import logging

from .base_model import BaseForecaster

logger = logging.getLogger(__name__)


class EnsemblePredictor(BaseForecaster):
    """Ensemble predictor that combines multiple forecasting models."""
    
    def __init__(self, models: Optional[Dict[str, BaseForecaster]] = None,
                 combination_method: str = 'average',
                 meta_learner: Optional[BaseEstimator] = None,
                 weights: Optional[Dict[str, float]] = None,
                 **kwargs):
        """Initialize ensemble predictor.
        
        Args:
            models: Dictionary of model name -> model instance
            combination_method: Method to combine predictions ('average', 'weighted', 'stacking')
            meta_learner: Meta-learner for stacking (if None, uses LinearRegression)
            weights: Weights for weighted averaging
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)
        
        self.models = models or {}
        self.combination_method = combination_method
        self.meta_learner = meta_learner
        self.weights = weights or {}
        
        # Initialize meta-learner for stacking
        if self.combination_method == 'stacking' and self.meta_learner is None:
            self.meta_learner = LinearRegression()
        
        # Track model performance for dynamic weighting
        self.model_performance = {}
        self.fitted_meta_learner = None
        
    def add_model(self, name: str, model: BaseForecaster, weight: float = 1.0):
        """Add a model to the ensemble.
        
        Args:
            name: Model name
            model: Model instance
            weight: Weight for weighted averaging
        """
        self.models[name] = model
        self.weights[name] = weight
        logger.info(f"Added model '{name}' to ensemble with weight {weight}")
    
    def remove_model(self, name: str):
        """Remove a model from the ensemble.
        
        Args:
            name: Model name to remove
        """
        if name in self.models:
            del self.models[name]
            if name in self.weights:
                del self.weights[name]
            if name in self.model_performance:
                del self.model_performance[name]
            logger.info(f"Removed model '{name}' from ensemble")
        else:
            logger.warning(f"Model '{name}' not found in ensemble")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsemblePredictor':
        """Fit all models in the ensemble.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self
        """
        if not self.models:
            raise ValueError("No models in ensemble. Add models before fitting.")
        
        logger.info(f"Fitting ensemble with {len(self.models)} models")
        
        # Fit individual models
        fitted_models = {}
        model_predictions = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Fitting model: {name}")
                fitted_model = model.fit(X, y)
                fitted_models[name] = fitted_model
                
                # Get predictions for meta-learning
                if self.combination_method == 'stacking':
                    predictions = fitted_model.predict(X)
                    model_predictions[name] = predictions
                
                # Calculate performance metrics
                train_predictions = fitted_model.predict(X)
                mse = mean_squared_error(y, train_predictions)
                self.model_performance[name] = {'mse': mse, 'rmse': np.sqrt(mse)}
                
                logger.info(f"Model {name} fitted successfully (RMSE: {np.sqrt(mse):.4f})")
                
            except Exception as e:
                logger.error(f"Failed to fit model {name}: {e}")
                # Remove failed model
                if name in fitted_models:
                    del fitted_models[name]
        
        self.models = fitted_models
        
        # Fit meta-learner for stacking
        if self.combination_method == 'stacking' and model_predictions:
            logger.info("Fitting meta-learner for stacking")
            
            # Create meta-features matrix
            meta_X = np.column_stack([model_predictions[name] for name in sorted(model_predictions.keys())])
            
            try:
                self.fitted_meta_learner = self.meta_learner.fit(meta_X, y)
                logger.info("Meta-learner fitted successfully")
            except Exception as e:
                logger.error(f"Failed to fit meta-learner: {e}")
                # Fall back to average combination
                self.combination_method = 'average'
        
        # Update weights based on performance if using dynamic weighting
        if self.combination_method == 'weighted' and not self.weights:
            self._calculate_performance_weights()
        
        self.is_fitted = True
        logger.info("Ensemble fitting completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble.
        
        Args:
            X: Features for prediction
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        if not self.models:
            raise ValueError("No fitted models in ensemble")
        
        # Get predictions from all models
        model_predictions = {}
        
        for name, model in self.models.items():
            try:
                predictions = model.predict(X)
                model_predictions[name] = predictions
            except Exception as e:
                logger.error(f"Failed to get predictions from model {name}: {e}")
        
        if not model_predictions:
            raise ValueError("No models produced valid predictions")
        
        # Combine predictions
        if self.combination_method == 'average':
            ensemble_predictions = self._average_predictions(model_predictions)
        elif self.combination_method == 'weighted':
            ensemble_predictions = self._weighted_predictions(model_predictions)
        elif self.combination_method == 'stacking':
            ensemble_predictions = self._stacking_predictions(model_predictions, X)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        return ensemble_predictions
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates.
        
        Args:
            X: Features for prediction
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions and uncertainties from all models
        model_predictions = {}
        model_uncertainties = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_with_uncertainty'):
                    pred, unc = model.predict_with_uncertainty(X)
                else:
                    pred = model.predict(X)
                    unc = np.full_like(pred, 0.1 * np.std(pred))  # Simple uncertainty estimate
                
                model_predictions[name] = pred
                model_uncertainties[name] = unc
                
            except Exception as e:
                logger.error(f"Failed to get predictions from model {name}: {e}")
        
        if not model_predictions:
            raise ValueError("No models produced valid predictions")
        
        # Combine predictions
        ensemble_predictions = self.predict(X)
        
        # Combine uncertainties
        if self.combination_method == 'average':
            # Average uncertainties and add prediction variance
            avg_uncertainty = np.mean([model_uncertainties[name] for name in model_uncertainties], axis=0)
            pred_variance = np.var([model_predictions[name] for name in model_predictions], axis=0)
            ensemble_uncertainties = np.sqrt(avg_uncertainty**2 + pred_variance)
            
        elif self.combination_method == 'weighted':
            # Weighted average of uncertainties
            total_weight = sum(self.weights.get(name, 1.0) for name in model_predictions)
            weighted_uncertainty = np.zeros_like(ensemble_predictions)
            
            for name in model_predictions:
                weight = self.weights.get(name, 1.0) / total_weight
                weighted_uncertainty += weight * model_uncertainties[name]**2
            
            ensemble_uncertainties = np.sqrt(weighted_uncertainty)
            
        else:
            # For stacking, use prediction variance as uncertainty
            pred_variance = np.var([model_predictions[name] for name in model_predictions], axis=0)
            ensemble_uncertainties = np.sqrt(pred_variance)
        
        return ensemble_predictions, ensemble_uncertainties
    
    def _average_predictions(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Average predictions from all models."""
        predictions_array = np.array([pred for pred in model_predictions.values()])
        return np.mean(predictions_array, axis=0)
    
    def _weighted_predictions(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average of predictions."""
        total_weight = sum(self.weights.get(name, 1.0) for name in model_predictions)
        
        if total_weight == 0:
            return self._average_predictions(model_predictions)
        
        weighted_sum = np.zeros_like(next(iter(model_predictions.values())))
        
        for name, predictions in model_predictions.items():
            weight = self.weights.get(name, 1.0) / total_weight
            weighted_sum += weight * predictions
        
        return weighted_sum
    
    def _stacking_predictions(self, model_predictions: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
        """Stacking predictions using meta-learner."""
        if self.fitted_meta_learner is None:
            logger.warning("Meta-learner not fitted, falling back to average")
            return self._average_predictions(model_predictions)
        
        # Create meta-features matrix
        meta_X = np.column_stack([model_predictions[name] for name in sorted(model_predictions.keys())])
        
        try:
            return self.fitted_meta_learner.predict(meta_X)
        except Exception as e:
            logger.error(f"Meta-learner prediction failed: {e}, falling back to average")
            return self._average_predictions(model_predictions)
    
    def _calculate_performance_weights(self):
        """Calculate weights based on model performance (inverse of RMSE)."""
        if not self.model_performance:
            return
        
        # Calculate inverse RMSE weights
        rmse_values = [perf['rmse'] for perf in self.model_performance.values()]
        inv_rmse = [1.0 / rmse if rmse > 0 else 1.0 for rmse in rmse_values]
        total_inv_rmse = sum(inv_rmse)
        
        if total_inv_rmse > 0:
            for i, name in enumerate(self.model_performance.keys()):
                self.weights[name] = inv_rmse[i] / total_inv_rmse
        
        logger.info(f"Updated performance-based weights: {self.weights}")
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all models."""
        return self.model_performance.copy()
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weights.copy()
    
    def set_weights(self, weights: Dict[str, float]):
        """Set model weights for weighted averaging.
        
        Args:
            weights: Dictionary of model name -> weight
        """
        self.weights = weights.copy()
        logger.info(f"Updated model weights: {self.weights}")
    
    def get_model_names(self) -> List[str]:
        """Get names of all models in the ensemble."""
        return list(self.models.keys())
