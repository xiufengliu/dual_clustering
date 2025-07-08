"""Tests for the neutrosophic forecasting framework."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.framework.forecasting_framework import NeutrosophicForecastingFramework
from src.clustering.dual_clusterer import DualClusterer
from src.neutrosophic.neutrosophic_transformer import NeutrosophicTransformer
from src.models.random_forest_model import RandomForestForecaster
from src.evaluation.metrics import ForecastingMetrics


class TestNeutrosophicFramework:
    """Test suite for the neutrosophic forecasting framework."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample renewable energy data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        # Generate synthetic time series with daily pattern
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='H')
        
        # Simple daily pattern for solar-like data
        hours = np.array([ts.hour for ts in timestamps])
        daily_pattern = np.maximum(0, np.sin((hours - 6) * np.pi / 12))
        
        # Add noise and trend
        noise = np.random.normal(0, 0.1, n_samples)
        trend = np.linspace(0, 0.2, n_samples)
        
        energy_generation = 50 + 30 * daily_pattern + 10 * trend + 5 * noise
        energy_generation = np.maximum(0, energy_generation)  # Ensure non-negative
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'energy_generation': energy_generation
        })
    
    @pytest.fixture
    def framework_config(self):
        """Create test configuration for the framework."""
        return {
            'clustering': {
                'n_clusters': 3,
                'fcm_fuzziness': 2.0,
                'max_iter': 50,
                'tol': 1e-3,
                'random_state': 42
            },
            'neutrosophic': {
                'entropy_epsilon': 1e-9,
                'entropy_base': 2.0
            },
            'random_forest': {
                'n_estimators': 10,  # Small for testing
                'max_depth': 5,
                'random_state': 42
            },
            'forecasting': {
                'horizon': 5,
                'gamma': 1.0,
                'beta': 0.5
            },
            'data': {
                'normalization': 'min_max',
                'train_split': 0.8
            },
            'reproducibility': {
                'seed': 42
            }
        }
    
    def test_framework_initialization(self, framework_config):
        """Test framework initialization."""
        framework = NeutrosophicForecastingFramework(config=framework_config)
        
        assert framework.config == framework_config
        assert not framework.is_fitted
        assert framework.dual_clusterer is not None
        assert framework.neutrosophic_transformer is not None
        assert framework.rf_model is not None
    
    def test_framework_fit(self, sample_data, framework_config):
        """Test framework fitting process."""
        framework = NeutrosophicForecastingFramework(config=framework_config)
        
        # Fit the framework
        framework.fit(sample_data)
        
        assert framework.is_fitted
        assert framework.preprocessing_params is not None
        assert framework.feature_names is not None
        assert framework.neutrosophic_components is not None
        assert framework.training_data is not None
    
    def test_framework_predict(self, sample_data, framework_config):
        """Test framework prediction."""
        framework = NeutrosophicForecastingFramework(config=framework_config)
        
        # Fit and predict
        framework.fit(sample_data)
        
        horizon = 3
        predictions = framework.predict(horizon=horizon, return_intervals=True)
        
        assert 'predictions' in predictions
        assert 'lower_bounds' in predictions
        assert 'upper_bounds' in predictions
        assert len(predictions['predictions']) == horizon
        assert len(predictions['lower_bounds']) == horizon
        assert len(predictions['upper_bounds']) == horizon
        
        # Check that intervals are valid
        assert np.all(predictions['lower_bounds'] <= predictions['upper_bounds'])
    
    def test_dual_clusterer(self):
        """Test dual clustering component."""
        np.random.seed(42)
        X = np.random.rand(50, 1)
        
        clusterer = DualClusterer(n_clusters=3, random_state=42)
        clusterer.fit(X)
        
        assert clusterer.is_fitted
        
        # Test predictions
        kmeans_labels, fcm_memberships = clusterer.predict(X)
        assert len(kmeans_labels) == len(X)
        assert fcm_memberships.shape == (len(X), 3)
        
        # Test integrated features
        integrated_features = clusterer.get_integrated_features()
        assert integrated_features.shape == (len(X), 6)  # 3 one-hot + 3 FCM
    
    def test_neutrosophic_transformer(self):
        """Test neutrosophic transformation."""
        np.random.seed(42)
        n_samples = 20
        n_clusters = 3
        
        # Create sample clustering outputs
        kmeans_labels = np.random.randint(0, n_clusters, n_samples)
        fcm_memberships = np.random.dirichlet([1] * n_clusters, n_samples)
        
        transformer = NeutrosophicTransformer()
        components = transformer.transform(kmeans_labels, fcm_memberships)
        
        assert len(components.truth) == n_samples
        assert len(components.indeterminacy) == n_samples
        assert len(components.falsity) == n_samples
        
        # Check value ranges
        assert np.all(components.truth >= 0) and np.all(components.truth <= 1)
        assert np.all(components.indeterminacy >= 0) and np.all(components.indeterminacy <= 1)
        assert np.all(components.falsity >= 0) and np.all(components.falsity <= 1)
        
        # Check T + F = 1 (approximately)
        assert np.allclose(components.truth + components.falsity, 1.0)
    
    def test_random_forest_model(self):
        """Test Random Forest model component."""
        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = np.random.rand(50)
        
        model = RandomForestForecaster(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        assert model.is_fitted
        
        # Test predictions
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        
        # Test uncertainty predictions
        pred_mean, pred_std = model.predict_with_uncertainty(X)
        assert len(pred_mean) == len(X)
        assert len(pred_std) == len(X)
        assert np.all(pred_std >= 0)  # Standard deviation should be non-negative
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics."""
        np.random.seed(42)
        n_samples = 30
        
        y_true = np.random.rand(n_samples) * 100
        y_pred = y_true + np.random.normal(0, 5, n_samples)  # Add some error
        
        lower_bounds = y_pred - 10
        upper_bounds = y_pred + 10
        
        metrics = ForecastingMetrics()
        
        # Test point metrics
        point_metrics = metrics.calculate_point_metrics(y_true, y_pred)
        
        assert 'rmse' in point_metrics
        assert 'mae' in point_metrics
        assert 'mape' in point_metrics
        assert 'r2' in point_metrics
        
        assert point_metrics['rmse'] >= 0
        assert point_metrics['mae'] >= 0
        assert point_metrics['mape'] >= 0
        
        # Test interval metrics
        interval_metrics = metrics.calculate_interval_metrics(
            y_true, lower_bounds, upper_bounds, confidence_level=0.95
        )
        
        assert 'picp' in interval_metrics
        assert 'pinaw' in interval_metrics
        assert 'ace' in interval_metrics
        
        assert 0 <= interval_metrics['picp'] <= 1
        assert interval_metrics['pinaw'] >= 0
        assert interval_metrics['ace'] >= 0
    
    def test_framework_save_load(self, sample_data, framework_config, tmp_path):
        """Test framework save and load functionality."""
        framework = NeutrosophicForecastingFramework(config=framework_config)
        framework.fit(sample_data)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        framework.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_framework = NeutrosophicForecastingFramework.load_model(str(model_path))
        
        assert loaded_framework.is_fitted
        assert loaded_framework.config == framework_config
        
        # Test that loaded model can make predictions
        predictions = loaded_framework.predict(horizon=2)
        assert 'predictions' in predictions
        assert len(predictions['predictions']) == 2
    
    def test_framework_feature_importance(self, sample_data, framework_config):
        """Test feature importance analysis."""
        framework = NeutrosophicForecastingFramework(config=framework_config)
        framework.fit(sample_data)
        
        feature_importance = framework.get_feature_importance()
        
        assert 'feature_importance_ranking' in feature_importance
        assert 'neutrosophic_analysis' in feature_importance
        assert 'feature_names' in feature_importance
        
        # Check that neutrosophic features are included
        neutrosophic_analysis = feature_importance['neutrosophic_analysis']
        if neutrosophic_analysis:  # May be empty if no neutrosophic features found
            assert 'total_neutrosophic_importance' in neutrosophic_analysis
    
    def test_error_handling(self, framework_config):
        """Test error handling in various scenarios."""
        framework = NeutrosophicForecastingFramework(config=framework_config)
        
        # Test prediction before fitting
        with pytest.raises(ValueError, match="must be fitted"):
            framework.predict(horizon=1)
        
        # Test invalid data
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        
        with pytest.raises((ValueError, KeyError)):
            framework.fit(invalid_data)


if __name__ == "__main__":
    pytest.main([__file__])