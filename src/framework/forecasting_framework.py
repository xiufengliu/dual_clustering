"""Main neutrosophic forecasting framework implementation."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path

from ..data.data_loader import ENTSOEDataLoader
from ..data.preprocessor import DataPreprocessor
from ..data.validator import DataValidator
from ..clustering.dual_clusterer import DualClusterer
from ..neutrosophic.neutrosophic_transformer import NeutrosophicTransformer, NeutrosophicComponents
from ..models.random_forest_model import RandomForestForecaster
from ..utils.config_manager import ConfigManager
from ..utils.logger import setup_logger
from ..utils.math_utils import set_random_seeds

logger = logging.getLogger(__name__)


class NeutrosophicForecastingFramework:
    """
    Main framework implementing the neutrosophic dual clustering approach.
    
    This class orchestrates the complete pipeline from Algorithm 1 in the paper:
    1. Preprocessing
    2. Dual Clustering (K-Means + FCM)
    3. Neutrosophic Transformation
    4. Random Forest Training
    5. Prediction with Uncertainty Quantification
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 config_path: Optional[str] = None,
                 experiment_name: Optional[str] = None):
        """Initialize the neutrosophic forecasting framework.
        
        Args:
            config: Configuration dictionary
            config_path: Path to configuration file
            experiment_name: Name of experiment configuration
        """
        # Load configuration
        if config is not None:
            self.config = config
        else:
            config_manager = ConfigManager()
            self.config = config_manager.get_config(experiment_name=experiment_name)
        
        # Set random seeds for reproducibility
        seed = self.config.get('reproducibility', {}).get('seed', 42)
        set_random_seeds(seed)
        
        # Initialize components
        self.data_loader = None
        self.data_validator = DataValidator(self.config.get('data', {}))
        self.preprocessor = DataPreprocessor(self.config.get('data', {}))
        self.dual_clusterer = DualClusterer(**self.config.get('clustering', {}))
        self.neutrosophic_transformer = NeutrosophicTransformer(**self.config.get('neutrosophic', {}))
        self.rf_model = RandomForestForecaster(**self.config.get('random_forest', {}))
        
        # Framework state
        self.is_fitted = False
        self.training_data = None
        self.preprocessing_params = None
        self.feature_names = None
        self.neutrosophic_components = None
        
        # Setup logging
        log_config = self.config.get('logging', {})
        self.logger = setup_logger(
            name=self.__class__.__name__,
            level=getattr(logging, log_config.get('level', 'INFO'))
        )
        
        self.logger.info("Neutrosophic Forecasting Framework initialized")
    
    def load_data(self, dataset_type: str = "solar", **kwargs) -> pd.DataFrame:
        """Load renewable energy data.
        
        Args:
            dataset_type: Type of dataset ('solar' or 'wind')
            **kwargs: Additional arguments for data loading
            
        Returns:
            Loaded DataFrame
        """
        self.logger.info(f"Loading {dataset_type} energy data")
        
        # Initialize data loader if not already done
        if self.data_loader is None:
            self.data_loader = ENTSOEDataLoader()
        
        # Load data based on type
        if dataset_type == "solar":
            data = self.data_loader.load_solar_data(**kwargs)
        elif dataset_type == "wind":
            data = self.data_loader.load_wind_data(**kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Validate data
        is_valid, validation_report = self.data_validator.validate_dataset(data)
        if not is_valid:
            raise ValueError(f"Data validation failed: {validation_report['errors']}")
        
        self.logger.info(f"Data loaded successfully: {len(data)} samples")
        return data
    
    def fit(self, data: pd.DataFrame, target_column: str = 'energy_generation') -> 'NeutrosophicForecastingFramework':
        """Fit the complete neutrosophic forecasting framework.
        
        Implementation of Algorithm 1 from the paper.
        
        Args:
            data: Input DataFrame with time series data
            target_column: Name of target column
            
        Returns:
            Self
        """
        self.logger.info("Starting framework training")
        
        # Stage 1: Preprocessing
        self.logger.info("Stage 1: Data preprocessing")
        normalized_data, self.preprocessing_params = self.preprocessor.preprocess(data, fit=True)
        
        # For this implementation, we use the normalized time series as features
        # In practice, you might want to create lag features, time features, etc.
        X = normalized_data.reshape(-1, 1)  # Single feature: normalized energy generation
        y = normalized_data  # Target is the same (for autoregressive forecasting)
        
        # Stage 2: Dual Clustering
        self.logger.info("Stage 2: Dual clustering")
        self.dual_clusterer.fit(X)
        
        # Get integrated cluster features
        integrated_features = self.dual_clusterer.get_integrated_features()
        
        # Stage 3: Neutrosophic Transformation
        self.logger.info("Stage 3: Neutrosophic transformation")
        kmeans_labels, fcm_memberships = self.dual_clusterer.get_cluster_assignments()
        self.neutrosophic_components = self.neutrosophic_transformer.transform(
            kmeans_labels, fcm_memberships
        )
        
        # Create enriched feature set
        enriched_features = self.neutrosophic_transformer.create_enriched_features(
            X, integrated_features, self.neutrosophic_components
        )
        
        # Generate feature names
        n_clusters = self.config.get('clustering', {}).get('n_clusters', 5)
        self.feature_names = self.neutrosophic_transformer.get_feature_names(
            ['normalized_energy'], n_clusters
        )
        
        # Stage 4: Random Forest Training
        self.logger.info("Stage 4: Random Forest training")
        self.rf_model.fit(enriched_features, y)
        
        # Store training data for future reference
        self.training_data = {
            'original_data': data,
            'normalized_data': normalized_data,
            'enriched_features': enriched_features,
            'target': y
        }
        
        self.is_fitted = True
        self.logger.info("Framework training completed successfully")
        
        return self
    
    def predict(self, data: Optional[pd.DataFrame] = None, 
                horizon: int = 1, 
                return_intervals: bool = True,
                confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """Make predictions using the fitted framework.
        
        Args:
            data: Input data for prediction (if None, uses last training point)
            horizon: Forecast horizon
            return_intervals: Whether to return prediction intervals
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with predictions and optional intervals
        """
        if not self.is_fitted:
            raise ValueError("Framework must be fitted before prediction")
        
        self.logger.info(f"Making predictions with horizon {horizon}")
        
        # Use last training point if no data provided
        if data is None:
            if self.training_data is None:
                raise ValueError("No training data available for prediction")
            
            # Use the last point from training data
            last_normalized = self.training_data['normalized_data'][-1:]
            X_input = last_normalized.reshape(-1, 1)
        else:
            # Preprocess new data
            normalized_data, _ = self.preprocessor.preprocess(data, fit=False)
            X_input = normalized_data.reshape(-1, 1)
        
        # Apply the same transformation pipeline
        predictions_list = []
        intervals_list = []
        
        current_input = X_input[-1:].copy()  # Start with last available point
        
        for step in range(horizon):
            # Apply dual clustering
            kmeans_labels, fcm_memberships = self.dual_clusterer.predict(current_input)
            
            # Apply neutrosophic transformation
            neutrosophic_components = self.neutrosophic_transformer.transform(
                kmeans_labels, fcm_memberships
            )
            
            # Get integrated features
            integrated_features = self.dual_clusterer.get_integrated_features()
            
            # Create enriched features
            enriched_features = self.neutrosophic_transformer.create_enriched_features(
                current_input, integrated_features[-1:], neutrosophic_components
            )
            
            # Make prediction
            if return_intervals:
                # Get prediction intervals using neutrosophic indeterminacy
                gamma = self.config.get('forecasting', {}).get('gamma', 1.96)
                beta = self.config.get('forecasting', {}).get('beta', 1.0)
                
                pred, lower, upper = self.rf_model.predict_intervals_with_neutrosophic(
                    enriched_features, 
                    neutrosophic_components.indeterminacy,
                    confidence_level=confidence_level,
                    gamma=gamma,
                    beta=beta
                )
                
                predictions_list.append(pred[0])
                intervals_list.append((lower[0], upper[0]))
            else:
                pred = self.rf_model.predict(enriched_features)
                predictions_list.append(pred[0])
            
            # Update input for next step (recursive forecasting)
            current_input = np.array([[pred[0]]])
        
        # Convert to arrays
        predictions = np.array(predictions_list)
        
        # Denormalize predictions
        denormalized_predictions = self.preprocessor.inverse_transform(predictions)
        
        results = {
            'predictions': denormalized_predictions,
            'normalized_predictions': predictions
        }
        
        if return_intervals:
            lower_bounds = np.array([interval[0] for interval in intervals_list])
            upper_bounds = np.array([interval[1] for interval in intervals_list])
            
            # Denormalize intervals
            denormalized_lower = self.preprocessor.inverse_transform(lower_bounds)
            denormalized_upper = self.preprocessor.inverse_transform(upper_bounds)
            
            results.update({
                'lower_bounds': denormalized_lower,
                'upper_bounds': denormalized_upper,
                'normalized_lower_bounds': lower_bounds,
                'normalized_upper_bounds': upper_bounds,
                'confidence_level': confidence_level
            })
        
        self.logger.info(f"Predictions completed for horizon {horizon}")
        return results
    
    def evaluate(self, test_data: pd.DataFrame, 
                 target_column: str = 'energy_generation',
                 horizon: int = 1) -> Dict[str, Any]:
        """Evaluate the framework on test data.
        
        Args:
            test_data: Test DataFrame
            target_column: Name of target column
            horizon: Forecast horizon
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Framework must be fitted before evaluation")
        
        self.logger.info("Evaluating framework performance")
        
        # Make predictions
        predictions_dict = self.predict(test_data, horizon=horizon, return_intervals=True)
        predictions = predictions_dict['predictions']
        
        # Get true values (simplified - in practice you'd need proper test setup)
        true_values = test_data[target_column].values[:len(predictions)]
        
        # Calculate metrics
        from ..evaluation.metrics import ForecastingMetrics
        metrics_calculator = ForecastingMetrics()
        
        point_metrics = metrics_calculator.calculate_point_metrics(true_values, predictions)
        
        evaluation_results = {
            'point_metrics': point_metrics,
            'n_predictions': len(predictions),
            'horizon': horizon
        }
        
        # Add interval metrics if available
        if 'lower_bounds' in predictions_dict:
            interval_metrics = metrics_calculator.calculate_interval_metrics(
                true_values, 
                predictions_dict['lower_bounds'], 
                predictions_dict['upper_bounds'],
                predictions_dict['confidence_level']
            )
            evaluation_results['interval_metrics'] = interval_metrics
        
        self.logger.info("Evaluation completed")
        return evaluation_results
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance analysis.
        
        Returns:
            Dictionary with feature importance information
        """
        if not self.is_fitted:
            raise ValueError("Framework must be fitted before accessing feature importance")
        
        # Get feature importance from Random Forest
        importance_ranking = self.rf_model.get_feature_importance_ranking(self.feature_names)
        
        # Analyze neutrosophic feature importance
        neutrosophic_analysis = self.rf_model.analyze_neutrosophic_feature_importance(self.feature_names)
        
        return {
            'feature_importance_ranking': importance_ranking,
            'neutrosophic_analysis': neutrosophic_analysis,
            'feature_names': self.feature_names
        }
    
    def get_framework_info(self) -> Dict[str, Any]:
        """Get comprehensive framework information.
        
        Returns:
            Dictionary with framework information
        """
        info = {
            'is_fitted': self.is_fitted,
            'config': self.config,
            'components': {
                'data_validator': str(self.data_validator),
                'preprocessor': str(self.preprocessor),
                'dual_clusterer': str(self.dual_clusterer),
                'neutrosophic_transformer': str(self.neutrosophic_transformer),
                'rf_model': str(self.rf_model)
            }
        }
        
        if self.is_fitted:
            info.update({
                'feature_names': self.feature_names,
                'n_features': len(self.feature_names) if self.feature_names else None,
                'preprocessing_params': self.preprocessing_params
            })
            
            # Add clustering info
            if hasattr(self.dual_clusterer, 'get_comprehensive_info'):
                info['clustering_info'] = self.dual_clusterer.get_comprehensive_info()
            
            # Add neutrosophic analysis
            if self.neutrosophic_components:
                from ..neutrosophic.uncertainty_quantifier import UncertaintyQuantifier
                uncertainty_quantifier = UncertaintyQuantifier()
                info['neutrosophic_analysis'] = uncertainty_quantifier.quantify_uncertainty(
                    self.neutrosophic_components
                )
        
        return info
    
    def save_model(self, filepath: str) -> None:
        """Save the fitted framework to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Framework must be fitted before saving")
        
        import pickle
        
        save_data = {
            'config': self.config,
            'preprocessing_params': self.preprocessing_params,
            'dual_clusterer': self.dual_clusterer,
            'neutrosophic_transformer': self.neutrosophic_transformer,
            'rf_model': self.rf_model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"Framework saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'NeutrosophicForecastingFramework':
        """Load a fitted framework from file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded framework instance
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create new instance
        framework = cls(config=save_data['config'])
        
        # Restore state
        framework.preprocessing_params = save_data['preprocessing_params']
        framework.dual_clusterer = save_data['dual_clusterer']
        framework.neutrosophic_transformer = save_data['neutrosophic_transformer']
        framework.rf_model = save_data['rf_model']
        framework.feature_names = save_data['feature_names']
        framework.is_fitted = save_data['is_fitted']
        
        framework.logger.info(f"Framework loaded from {filepath}")
        return framework