"""Random Forest model implementation for neutrosophic forecasting framework."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Dict, Any, Optional, List
import logging

from .base_model import BaseForecaster

logger = logging.getLogger(__name__)


class RandomForestForecaster(BaseForecaster):
    """Random Forest forecaster with uncertainty quantification capabilities."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', bootstrap: bool = True,
                 random_state: Optional[int] = None, n_jobs: int = -1,
                 **kwargs):
        """Initialize Random Forest forecaster.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap sampling
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
            **kwargs: Additional arguments for RandomForestRegressor
        """
        super().__init__(random_state)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        
        # Initialize sklearn RandomForestRegressor
        self.rf_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **kwargs
        )
        
        # Store individual tree predictions for uncertainty estimation
        self.tree_predictions_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestForecaster':
        """Fit Random Forest to training data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            Self
        """
        self.validate_input(X, y)
        
        logger.info(f"Fitting Random Forest with {self.n_estimators} trees on data shape {X.shape}")
        
        # Fit the model
        self.rf_model.fit(X, y)
        
        # Store model information
        self.n_features_ = X.shape[1]
        self.is_fitted = True
        
        # Calculate training score
        train_score = self.rf_model.score(X, y)
        logger.info(f"Random Forest training completed. R² score: {train_score:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make point predictions using Random Forest.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("RandomForestForecaster must be fitted before prediction")
        
        self.validate_input(X)
        
        return self.rf_model.predict(X)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates based on tree variance.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_fitted:
            raise ValueError("RandomForestForecaster must be fitted before prediction")
        
        self.validate_input(X)
        
        # Get predictions from all individual trees
        tree_predictions = self._get_tree_predictions(X)
        
        # Calculate mean and standard deviation across trees
        predictions = np.mean(tree_predictions, axis=1)
        uncertainties = np.std(tree_predictions, axis=1)
        
        return predictions, uncertainties
    
    def predict_intervals_with_neutrosophic(self, X: np.ndarray, 
                                          indeterminacy: np.ndarray,
                                          confidence_level: float = 0.95,
                                          gamma: float = 1.96, 
                                          beta: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with intervals incorporating neutrosophic indeterminacy.
        
        Implementation of Proposition 4 from the paper:
        Δ_{t+h} = γ * σ_{RF,t+h} + β * I_t
        
        Args:
            X: Feature matrix
            indeterminacy: Neutrosophic indeterminacy values
            confidence_level: Confidence level for intervals
            gamma: Weight for RF uncertainty
            beta: Weight for neutrosophic indeterminacy
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_fitted:
            raise ValueError("RandomForestForecaster must be fitted before prediction")
        
        self.validate_input(X)
        
        if len(indeterminacy) != len(X):
            raise ValueError("Indeterminacy array must have same length as X")
        
        # Get RF predictions and uncertainties
        predictions, rf_uncertainties = self.predict_with_uncertainty(X)
        
        # Calculate enhanced interval width using neutrosophic indeterminacy
        interval_widths = gamma * rf_uncertainties + beta * indeterminacy
        
        # Calculate bounds
        lower_bounds = predictions - interval_widths
        upper_bounds = predictions + interval_widths
        
        logger.info(f"Generated prediction intervals with neutrosophic enhancement")
        logger.info(f"Average RF uncertainty: {np.mean(rf_uncertainties):.4f}")
        logger.info(f"Average indeterminacy: {np.mean(indeterminacy):.4f}")
        logger.info(f"Average interval width: {np.mean(interval_widths):.4f}")
        
        return predictions, lower_bounds, upper_bounds
    
    def _get_tree_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all individual trees.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tree predictions of shape (n_samples, n_estimators)
        """
        tree_predictions = np.zeros((X.shape[0], self.n_estimators))
        
        for i, tree in enumerate(self.rf_model.estimators_):
            tree_predictions[:, i] = tree.predict(X)
        
        return tree_predictions
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores from Random Forest.
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted:
            raise ValueError("RandomForestForecaster must be fitted before accessing feature importance")
        
        return self.rf_model.feature_importances_
    
    def get_feature_importance_ranking(self, feature_names: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Get ranked feature importance with names.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        importances = self.get_feature_importance()
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        if len(feature_names) != len(importances):
            raise ValueError("Number of feature names must match number of features")
        
        # Create list of (name, importance) tuples and sort by importance
        importance_pairs = list(zip(feature_names, importances))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return importance_pairs
    
    def analyze_neutrosophic_feature_importance(self, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze importance of neutrosophic features specifically.
        
        Args:
            feature_names: List of all feature names
            
        Returns:
            Dictionary with neutrosophic feature analysis
        """
        if not self.is_fitted:
            raise ValueError("RandomForestForecaster must be fitted before analysis")
        
        importances = self.get_feature_importance()
        
        # Identify neutrosophic features
        neutrosophic_indices = []
        neutrosophic_names = []
        
        for i, name in enumerate(feature_names):
            if name in ['truth', 'indeterminacy', 'falsity']:
                neutrosophic_indices.append(i)
                neutrosophic_names.append(name)
        
        if not neutrosophic_indices:
            logger.warning("No neutrosophic features found in feature names")
            return {}
        
        # Extract neutrosophic importances
        neutrosophic_importances = importances[neutrosophic_indices]
        total_neutrosophic_importance = np.sum(neutrosophic_importances)
        
        # Calculate relative importance within neutrosophic components
        relative_importances = neutrosophic_importances / total_neutrosophic_importance
        
        analysis = {
            'neutrosophic_feature_names': neutrosophic_names,
            'neutrosophic_importances': neutrosophic_importances.tolist(),
            'relative_neutrosophic_importances': relative_importances.tolist(),
            'total_neutrosophic_importance': float(total_neutrosophic_importance),
            'neutrosophic_importance_ratio': float(total_neutrosophic_importance / np.sum(importances)),
            'most_important_neutrosophic': neutrosophic_names[np.argmax(neutrosophic_importances)],
            'least_important_neutrosophic': neutrosophic_names[np.argmin(neutrosophic_importances)]
        }
        
        return analysis
    
    def get_tree_diversity_metrics(self) -> Dict[str, float]:
        """Calculate diversity metrics for the Random Forest ensemble.
        
        Returns:
            Dictionary with diversity metrics
        """
        if not self.is_fitted:
            raise ValueError("RandomForestForecaster must be fitted before calculating diversity")
        
        # This would require access to training data to calculate properly
        # For now, return basic ensemble information
        metrics = {
            'n_estimators': self.n_estimators,
            'max_depth': self.rf_model.max_depth if self.rf_model.max_depth else -1,
            'n_features': self.n_features_,
            'max_features_used': self.rf_model.max_features_
        }
        
        return metrics
    
    def get_params(self) -> Dict[str, Any]:
        """Get Random Forest parameters.
        
        Returns:
            Dictionary of parameters
        """
        params = super().get_params()
        params.update({
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'n_jobs': self.n_jobs
        })
        return params
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            raise ValueError("RandomForestForecaster must be fitted before accessing model info")
        
        info = {
            'model_type': 'RandomForestRegressor',
            'is_fitted': self.is_fitted,
            'n_features': self.n_features_,
            'n_estimators': self.n_estimators,
            'parameters': self.get_params(),
            'sklearn_model': str(self.rf_model)
        }
        
        return info