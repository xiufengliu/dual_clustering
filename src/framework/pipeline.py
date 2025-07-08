"""Forecasting pipeline for end-to-end renewable energy forecasting."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
import joblib
from datetime import datetime

from ..data.data_loader import BaseDataLoader
from ..data.preprocessor import DataPreprocessor
from ..clustering.dual_clusterer import DualClusterer
from ..neutrosophic.neutrosophic_transformer import NeutrosophicTransformer
from ..models.random_forest_model import RandomForestForecaster
from ..evaluation.metrics import ForecastingMetrics
from ..utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class ForecastingPipeline:
    """End-to-end forecasting pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize forecasting pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.data_loader = None
        self.preprocessor = DataPreprocessor()
        self.clusterer = DualClusterer()
        self.transformer = NeutrosophicTransformer()
        self.model = RandomForestForecaster()
        self.metrics = ForecastingMetrics()
        
        # Pipeline state
        self.is_fitted = False
        self.feature_names = None
        self.target_scaler = None
        
        # Results storage
        self.training_history = {}
        self.validation_results = {}
        
    def load_data(self, data_loader: BaseDataLoader, **kwargs) -> pd.DataFrame:
        """Load data using the provided data loader.
        
        Args:
            data_loader: Data loader instance
            **kwargs: Arguments for data loading
            
        Returns:
            Loaded data
        """
        self.data_loader = data_loader
        data = data_loader.load_data(**kwargs)
        
        logger.info(f"Loaded data: {len(data)} samples")
        return data
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data.
        
        Args:
            data: Raw data
            
        Returns:
            Preprocessed data
        """
        processed_data = self.preprocessor.preprocess(data)
        logger.info(f"Preprocessed data: {len(processed_data)} samples, {len(processed_data.columns)} features")
        return processed_data
    
    def create_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create features using dual clustering and neutrosophic transformation.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Tuple of (features, targets)
        """
        # Extract features and targets
        feature_columns = [col for col in data.columns if col != 'energy_generation']
        X_raw = data[feature_columns].values
        y = data['energy_generation'].values
        
        # Perform dual clustering
        cluster_labels, cluster_memberships = self.clusterer.fit_predict(X_raw)
        
        # Apply neutrosophic transformation
        neutrosophic_features = self.transformer.fit_transform(
            X_raw, cluster_labels, cluster_memberships
        )
        
        # Combine original features with neutrosophic features
        X_combined = np.hstack([X_raw, neutrosophic_features])
        
        # Store feature names
        original_names = feature_columns
        neutrosophic_names = self.transformer.get_feature_names()
        self.feature_names = original_names + neutrosophic_names
        
        logger.info(f"Created features: {X_combined.shape[1]} total features")
        logger.info(f"Original features: {len(original_names)}, Neutrosophic features: {len(neutrosophic_names)}")
        
        return X_combined, y
    
    def fit(self, data: pd.DataFrame, validation_split: float = 0.2) -> 'ForecastingPipeline':
        """Fit the complete pipeline.
        
        Args:
            data: Training data
            validation_split: Fraction of data to use for validation
            
        Returns:
            Self
        """
        logger.info("Starting pipeline fitting")
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Create train/validation split
        split_idx = int(len(processed_data) * (1 - validation_split))
        train_data = processed_data.iloc[:split_idx]
        val_data = processed_data.iloc[split_idx:]
        
        # Create features
        X_train, y_train = self.create_features(train_data)
        X_val, y_val = self.create_features(val_data)
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Validate on validation set
        val_predictions = self.model.predict(X_val)
        val_metrics = self.metrics.calculate_point_metrics(y_val, val_predictions)
        
        # Store results
        self.training_history = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'n_features': X_train.shape[1],
            'feature_names': self.feature_names
        }
        
        self.validation_results = val_metrics
        
        self.is_fitted = True
        
        logger.info("Pipeline fitting completed")
        logger.info(f"Validation RMSE: {val_metrics['rmse']:.4f}")
        
        return self
    
    def predict(self, data: pd.DataFrame, return_intervals: bool = False,
                confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """Make predictions using the fitted pipeline.
        
        Args:
            data: Data for prediction
            return_intervals: Whether to return prediction intervals
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with predictions and optionally intervals
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Create features
        X, _ = self.create_features(processed_data)
        
        # Make predictions
        if return_intervals and hasattr(self.model, 'predict_with_uncertainty'):
            predictions, uncertainties = self.model.predict_with_uncertainty(X)
            
            # Calculate prediction intervals
            alpha = 1 - confidence_level
            z_score = 1.96  # For 95% confidence (approximate)
            
            lower_bounds = predictions - z_score * uncertainties
            upper_bounds = predictions + z_score * uncertainties
            
            return {
                'predictions': predictions,
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'uncertainties': uncertainties
            }
        else:
            predictions = self.model.predict(X)
            return {'predictions': predictions}
    
    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the pipeline on test data.
        
        Args:
            data: Test data
            
        Returns:
            Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")
        
        # Get predictions
        results = self.predict(data, return_intervals=True)
        predictions = results['predictions']
        
        # Get true values
        processed_data = self.preprocess_data(data)
        y_true = processed_data['energy_generation'].values[:len(predictions)]
        
        # Calculate metrics
        point_metrics = self.metrics.calculate_point_metrics(y_true, predictions)
        
        # Calculate interval metrics if available
        if 'lower_bounds' in results and 'upper_bounds' in results:
            interval_metrics = self.metrics.calculate_interval_metrics(
                y_true, results['lower_bounds'], results['upper_bounds'], 0.95
            )
            return {**point_metrics, **interval_metrics}
        else:
            return point_metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the fitted model.
        
        Returns:
            Dictionary of feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before getting feature importance")
        
        if hasattr(self.model, 'get_feature_importance'):
            importance_scores = self.model.get_feature_importance()
            
            if self.feature_names and len(importance_scores) == len(self.feature_names):
                return dict(zip(self.feature_names, importance_scores))
            else:
                return {f'feature_{i}': score for i, score in enumerate(importance_scores)}
        else:
            logger.warning("Model does not support feature importance")
            return {}
    
    def save(self, filepath: Union[str, Path]):
        """Save the fitted pipeline.
        
        Args:
            filepath: Path to save the pipeline
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare pipeline state for saving
        pipeline_state = {
            'config': self.config,
            'preprocessor': self.preprocessor,
            'clusterer': self.clusterer,
            'transformer': self.transformer,
            'model': self.model,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'validation_results': self.validation_results,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(pipeline_state, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ForecastingPipeline':
        """Load a saved pipeline.
        
        Args:
            filepath: Path to the saved pipeline
            
        Returns:
            Loaded pipeline instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        
        pipeline_state = joblib.load(filepath)
        
        # Create new pipeline instance
        pipeline = cls(config=pipeline_state['config'])
        
        # Restore state
        pipeline.preprocessor = pipeline_state['preprocessor']
        pipeline.clusterer = pipeline_state['clusterer']
        pipeline.transformer = pipeline_state['transformer']
        pipeline.model = pipeline_state['model']
        pipeline.feature_names = pipeline_state['feature_names']
        pipeline.training_history = pipeline_state['training_history']
        pipeline.validation_results = pipeline_state['validation_results']
        pipeline.is_fitted = pipeline_state['is_fitted']
        
        logger.info(f"Pipeline loaded from {filepath}")
        return pipeline
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training process.
        
        Returns:
            Training summary
        """
        return {
            'training_history': self.training_history,
            'validation_results': self.validation_results,
            'is_fitted': self.is_fitted,
            'config': self.config
        }
