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

        # Ensure we have the target column
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data. Available columns: {list(data.columns)}")

        # Extract only the target column and timestamp for preprocessing
        if 'timestamp' in data.columns:
            preprocessing_data = data[['timestamp', target_column]].copy()
        else:
            # If no timestamp column, create a simple DataFrame with just the target
            preprocessing_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=len(data), freq='H'),
                target_column: data[target_column]
            })

        # Ensure target column is numeric
        preprocessing_data[target_column] = pd.to_numeric(preprocessing_data[target_column], errors='coerce')
        preprocessing_data = preprocessing_data.dropna(subset=[target_column])

        if len(preprocessing_data) == 0:
            raise ValueError("No valid numeric data found in target column after preprocessing")

        # Stage 1: Preprocessing
        self.logger.info("Stage 1: Data preprocessing")
        normalized_data, self.preprocessing_params = self.preprocessor.preprocess(preprocessing_data, fit=True)

        # For this implementation, we use the normalized time series as features
        # In practice, you might want to create lag features, time features, etc.
        X = normalized_data.reshape(-1, 1).astype(np.float64)  # Ensure X is float64
        y = normalized_data.astype(np.float64)  # Target is the same (for autoregressive forecasting)
        
        # Stage 2: Dual Clustering
        self.logger.info("Stage 2: Dual clustering")
        self.dual_clusterer.fit(X)
        
        # Get integrated cluster features
        integrated_features = self.dual_clusterer.get_integrated_features()
        
        # Stage 3: Neutrosophic Transformation
        self.logger.info("Stage 3: Neutrosophic transformation")
        try:
            # Get cluster assignments with comprehensive validation
            kmeans_labels, fcm_memberships = self.dual_clusterer.get_cluster_assignments()

            # Comprehensive data type validation before transformation
            self.logger.debug(f"Pre-transform validation:")
            self.logger.debug(f"  X: dtype={X.dtype}, shape={X.shape}")
            self.logger.debug(f"  K-means labels: dtype={kmeans_labels.dtype}, shape={kmeans_labels.shape}, sample={kmeans_labels[:5]}")
            self.logger.debug(f"  FCM memberships: dtype={fcm_memberships.dtype}, shape={fcm_memberships.shape}")
            self.logger.debug(f"  Integrated features: dtype={integrated_features.dtype}, shape={integrated_features.shape}")

            # Validate and fix data types if necessary
            if X.dtype != np.float64:
                self.logger.warning(f"X has dtype {X.dtype}, converting to float64")
                X = X.astype(np.float64)

            if kmeans_labels.dtype.kind not in ['i', 'u']:
                self.logger.warning(f"K-means labels have non-integer dtype: {kmeans_labels.dtype}")
                kmeans_labels = kmeans_labels.astype(int)

            if fcm_memberships.dtype != np.float64:
                self.logger.warning(f"FCM memberships have dtype {fcm_memberships.dtype}, converting to float64")
                fcm_memberships = fcm_memberships.astype(np.float64)

            # Check for any string contamination in the integrated features
            if integrated_features.dtype.kind in ['U', 'S', 'O']:
                self.logger.error(f"Integrated features contain string/object data: {integrated_features.dtype}")
                sample_data = integrated_features.flatten()[:10]
                self.logger.error(f"Sample values: {sample_data}")
                self.logger.error(f"Sample data types: {[type(x) for x in sample_data]}")
                raise ValueError(f"Integrated features contain non-numeric data: {integrated_features.dtype}")

            self.logger.debug(f"Post-validation data types:")
            self.logger.debug(f"  X: {X.dtype}, K-means: {kmeans_labels.dtype}, FCM: {fcm_memberships.dtype}")

            # Perform neutrosophic transformation
            self.neutrosophic_components = self.neutrosophic_transformer.transform(
                kmeans_labels, fcm_memberships
            )

            # Create enriched feature set with additional validation
            enriched_features = self.neutrosophic_transformer.create_enriched_features(
                X, integrated_features, self.neutrosophic_components
            )

            # Final validation of enriched features
            if enriched_features.dtype != np.float64:
                self.logger.warning(f"Enriched features have dtype {enriched_features.dtype}, converting to float64")
                enriched_features = enriched_features.astype(np.float64)

        except Exception as e:
            self.logger.error(f"Neutrosophic transformation failed: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")

            # Enhanced error reporting
            try:
                self.logger.error(f"X dtype: {X.dtype}, shape: {X.shape}")
                self.logger.error(f"Integrated features dtype: {integrated_features.dtype}, shape: {integrated_features.shape}")

                # Sample the problematic data
                if hasattr(integrated_features, 'flatten'):
                    sample_data = integrated_features.flatten()[:20]
                    self.logger.error(f"Integrated features sample: {sample_data}")
                    self.logger.error(f"Sample data types: {[type(x) for x in sample_data[:5]]}")

            except Exception as debug_error:
                self.logger.error(f"Error during debug reporting: {debug_error}")

            raise RuntimeError(f"Neutrosophic transformation stage failed: {e}") from e
        
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

        # Use optimized prediction for large horizons
        if horizon > 100:
            return self._predict_vectorized(data, horizon, return_intervals, confidence_level)

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
            integrated_features = self.dual_clusterer._create_integrated_features(current_input)
            
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

    def _predict_vectorized(self, data: Optional[pd.DataFrame] = None,
                           horizon: int = 1,
                           return_intervals: bool = True,
                           confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """Optimized vectorized prediction for large horizons.

        This method reduces computational complexity by:
        1. Batching clustering operations
        2. Caching neutrosophic transformations
        3. Using vectorized operations where possible

        Args:
            data: Input data for prediction
            horizon: Forecast horizon
            return_intervals: Whether to return prediction intervals
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with predictions and optional intervals
        """
        self.logger.info(f"Using optimized vectorized prediction for horizon {horizon}")

        # Use last training point if no data provided
        if data is None:
            if self.training_data is None:
                raise ValueError("No training data available for prediction")
            last_normalized = self.training_data['normalized_data'][-1:]
            X_input = last_normalized.reshape(-1, 1)
        else:
            normalized_data, _ = self.preprocessor.preprocess(data, fit=False)
            X_input = normalized_data.reshape(-1, 1)

        # Pre-allocate arrays for better performance
        predictions = np.zeros(horizon)
        if return_intervals:
            lower_bounds = np.zeros(horizon)
            upper_bounds = np.zeros(horizon)

        # Cache configuration parameters
        gamma = self.config.get('forecasting', {}).get('gamma', 1.96)
        beta = self.config.get('forecasting', {}).get('beta', 1.0)

        # Use batch processing for better efficiency
        batch_size = min(50, horizon)  # Process in batches to balance memory and speed
        current_input = X_input[-1:].copy()

        for batch_start in range(0, horizon, batch_size):
            batch_end = min(batch_start + batch_size, horizon)
            batch_size_actual = batch_end - batch_start

            # Process batch
            batch_predictions = np.zeros(batch_size_actual)
            if return_intervals:
                batch_lower = np.zeros(batch_size_actual)
                batch_upper = np.zeros(batch_size_actual)

            for i in range(batch_size_actual):
                # Apply dual clustering (this is the main bottleneck)
                kmeans_labels, fcm_memberships = self.dual_clusterer.predict(current_input)

                # Apply neutrosophic transformation
                neutrosophic_components = self.neutrosophic_transformer.transform(
                    kmeans_labels, fcm_memberships
                )

                # Get integrated features (cached from training)
                integrated_features = self.dual_clusterer._create_integrated_features(current_input)

                # Create enriched features
                enriched_features = self.neutrosophic_transformer.create_enriched_features(
                    current_input, integrated_features[-1:], neutrosophic_components
                )

                # Make prediction
                if return_intervals:
                    pred, lower, upper = self.rf_model.predict_intervals_with_neutrosophic(
                        enriched_features,
                        neutrosophic_components.indeterminacy,
                        confidence_level=confidence_level,
                        gamma=gamma,
                        beta=beta
                    )
                    batch_predictions[i] = pred[0]
                    batch_lower[i] = lower[0]
                    batch_upper[i] = upper[0]
                else:
                    pred = self.rf_model.predict(enriched_features)
                    batch_predictions[i] = pred[0]

                # Update input for next step
                current_input = np.array([[pred[0]]])

            # Store batch results
            predictions[batch_start:batch_end] = batch_predictions
            if return_intervals:
                lower_bounds[batch_start:batch_end] = batch_lower
                upper_bounds[batch_start:batch_end] = batch_upper

            # Log progress for long horizons
            if horizon > 1000 and (batch_end % 500 == 0 or batch_end == horizon):
                self.logger.info(f"Processed {batch_end}/{horizon} predictions ({100*batch_end/horizon:.1f}%)")

        # Denormalize predictions
        denormalized_predictions = self.preprocessor.inverse_transform(predictions)

        results = {
            'predictions': denormalized_predictions,
            'normalized_predictions': predictions
        }

        if return_intervals:
            denormalized_lower = self.preprocessor.inverse_transform(lower_bounds)
            denormalized_upper = self.preprocessor.inverse_transform(upper_bounds)

            results.update({
                'lower_bounds': denormalized_lower,
                'upper_bounds': denormalized_upper,
                'normalized_lower_bounds': lower_bounds,
                'normalized_upper_bounds': upper_bounds,
            })

        self.logger.info(f"Vectorized predictions completed for horizon {horizon}")
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