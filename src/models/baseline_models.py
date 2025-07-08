"""Comprehensive baseline models for renewable energy forecasting comparison."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("Statsmodels not available. Install with: pip install statsmodels")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")

from .base_model import BaseForecaster


class ARIMAForecaster(BaseForecaster):
    """ARIMA baseline forecaster."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.model = None
        self.fitted_values = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ARIMAForecaster':
        """Fit ARIMA model."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels required for ARIMA. Install with: pip install statsmodels")
            
        # ARIMA uses only the target series
        self.model = ARIMA(y, order=self.order)
        self.fitted_model = self.model.fit()
        self.fitted_values = self.fitted_model.fittedvalues
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        n_steps = len(X)
        forecast = self.fitted_model.forecast(steps=n_steps)
        return np.array(forecast)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        n_steps = len(X)
        forecast_result = self.fitted_model.get_forecast(steps=n_steps)

        # Handle both pandas Series and numpy arrays
        predictions = forecast_result.predicted_mean
        if hasattr(predictions, 'values'):
            predictions = predictions.values
        else:
            predictions = np.array(predictions)

        uncertainties = forecast_result.se_mean
        if hasattr(uncertainties, 'values'):
            uncertainties = uncertainties.values
        else:
            uncertainties = np.array(uncertainties)

        return predictions, uncertainties


class SARIMAForecaster(BaseForecaster):
    """Seasonal ARIMA baseline forecaster."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), 
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24), **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SARIMAForecaster':
        """Fit SARIMA model."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels required for SARIMA. Install with: pip install statsmodels")
            
        self.model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order)
        self.fitted_model = self.model.fit(disp=False)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        n_steps = len(X)
        forecast = self.fitted_model.forecast(steps=n_steps)
        return np.array(forecast)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        n_steps = len(X)
        forecast_result = self.fitted_model.get_forecast(steps=n_steps)

        # Handle both pandas Series and numpy arrays
        predictions = forecast_result.predicted_mean
        if hasattr(predictions, 'values'):
            predictions = predictions.values
        else:
            predictions = np.array(predictions)

        uncertainties = forecast_result.se_mean
        if hasattr(uncertainties, 'values'):
            uncertainties = uncertainties.values
        else:
            uncertainties = np.array(uncertainties)

        return predictions, uncertainties


class SVRForecaster(BaseForecaster):
    """Support Vector Regression baseline."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale', **kwargs):
        super().__init__(**kwargs)
        self.model = SVR(kernel=kernel, C=C, gamma=gamma)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVRForecaster':
        """Fit SVR model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        predictions = self.predict(X)
        # SVR doesn't provide uncertainty estimates, use constant uncertainty
        uncertainties = np.full_like(predictions, np.std(predictions) * 0.1)
        return predictions, uncertainties


class LightGBMForecaster(BaseForecaster):
    """LightGBM baseline forecaster."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, 
                 max_depth: int = -1, **kwargs):
        super().__init__(**kwargs)
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM required. Install with: pip install lightgbm")
            
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
            verbose=-1
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LightGBMForecaster':
        """Fit LightGBM model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        predictions = self.predict(X)
        # Use feature importance as proxy for uncertainty
        feature_importance = self.model.feature_importances_
        uncertainties = np.full_like(predictions, np.std(predictions) * 0.1)
        return predictions, uncertainties


class VanillaRandomForestForecaster(BaseForecaster):
    """Vanilla Random Forest baseline (without neutrosophic features)."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VanillaRandomForestForecaster':
        """Fit Random Forest model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        predictions = np.mean(tree_predictions, axis=0)
        uncertainties = np.std(tree_predictions, axis=0)
        
        return predictions, uncertainties


class MLPForecaster(BaseForecaster):
    """Multi-Layer Perceptron baseline."""

    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (100, 50),
                 activation: str = 'relu', max_iter: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=max_iter,
            random_state=self.random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPForecaster':
        """Fit MLP model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        predictions = self.predict(X)
        # MLP doesn't provide uncertainty estimates, use constant uncertainty
        uncertainties = np.full_like(predictions, np.std(predictions) * 0.1)
        return predictions, uncertainties


class LSTMNet(nn.Module):
    """PyTorch LSTM network."""

    def __init__(self, input_size: int, lstm_units: int = 50, dense_units: int = 25):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_units, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(lstm_units, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        x = self.dropout(lstm_out)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LSTMForecaster(BaseForecaster):
    """LSTM baseline forecaster using PyTorch."""

    def __init__(self, lstm_units: int = 50, dense_units: int = 25,
                 epochs: int = 100, batch_size: int = 32,
                 sequence_length: int = 24, **kwargs):
        super().__init__(**kwargs)
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for LSTM. Install with: pip install torch")

        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LSTMForecaster':
        """Fit LSTM model."""
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        input_size = X.shape[1]
        self.model = LSTMNet(input_size, self.lstm_units, self.dense_units).to(self.device)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Train model
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Create sequences for prediction
        if len(X) < self.sequence_length:
            # Pad with last known values if needed
            X_padded = np.vstack([X[0:1]] * (self.sequence_length - len(X)) + [X])
        else:
            X_padded = X

        X_seq, _ = self._create_sequences(X_padded, np.zeros(len(X_padded)))

        # Convert to tensor and predict
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()

        # Ensure predictions match expected length
        expected_length = len(X)
        if len(predictions) != expected_length:
            # Repeat last prediction or truncate to match expected length
            if len(predictions) < expected_length:
                predictions = np.pad(predictions, (0, expected_length - len(predictions)),
                                   mode='edge')
            else:
                predictions = predictions[:expected_length]

        return predictions

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        predictions = self.predict(X)
        # LSTM doesn't provide uncertainty estimates, use constant uncertainty
        uncertainties = np.full_like(predictions, np.std(predictions) * 0.1)
        return predictions, uncertainties


class CNNLSTMNet(nn.Module):
    """PyTorch CNN-LSTM network."""

    def __init__(self, input_size: int, cnn_filters: int = 64, kernel_size: int = 3,
                 lstm_units: int = 50, dense_units: int = 25):
        super(CNNLSTMNet, self).__init__()
        self.conv1d = nn.Conv1d(input_size, cnn_filters, kernel_size)
        self.maxpool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(cnn_filters, lstm_units, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(lstm_units, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Conv1d expects: (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1d(x))
        x = self.maxpool(x)
        # Back to (batch, seq_len, features) for LSTM
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        # Take the last output
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNLSTMForecaster(BaseForecaster):
    """CNN-LSTM hybrid baseline forecaster using PyTorch."""

    def __init__(self, cnn_filters: int = 64, kernel_size: int = 3,
                 lstm_units: int = 50, dense_units: int = 25,
                 epochs: int = 100, batch_size: int = 32,
                 sequence_length: int = 24, **kwargs):
        super().__init__(**kwargs)
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for CNN-LSTM. Install with: pip install torch")

        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for CNN-LSTM training."""
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CNNLSTMForecaster':
        """Fit CNN-LSTM model."""
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        input_size = X.shape[1]
        self.model = CNNLSTMNet(input_size, self.cnn_filters, self.kernel_size,
                               self.lstm_units, self.dense_units).to(self.device)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Train model
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Create sequences for prediction
        if len(X) < self.sequence_length:
            X_padded = np.vstack([X[0:1]] * (self.sequence_length - len(X)) + [X])
        else:
            X_padded = X

        X_seq, _ = self._create_sequences(X_padded, np.zeros(len(X_padded)))

        # Convert to tensor and predict
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()

        # Ensure predictions match expected length
        expected_length = len(X)
        if len(predictions) != expected_length:
            # Repeat last prediction or truncate to match expected length
            if len(predictions) < expected_length:
                predictions = np.pad(predictions, (0, expected_length - len(predictions)),
                                   mode='edge')
            else:
                predictions = predictions[:expected_length]

        return predictions

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        predictions = self.predict(X)
        uncertainties = np.full_like(predictions, np.std(predictions) * 0.1)
        return predictions, uncertainties


class NBeatsNet(nn.Module):
    """Simplified N-BEATS network using PyTorch."""

    def __init__(self, backcast_length: int, forecast_length: int, hidden_units: int = 256):
        super(NBeatsNet, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        # Simplified N-BEATS with just dense layers
        self.fc1 = nn.Linear(backcast_length, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, hidden_units)
        self.forecast_head = nn.Linear(hidden_units, forecast_length)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        forecast = self.forecast_head(x)
        return forecast


class NBeatsForecaster(BaseForecaster):
    """N-BEATS baseline forecaster (simplified implementation using PyTorch)."""

    def __init__(self, stack_types: List[str] = ['trend', 'seasonality'],
                 nb_blocks_per_stack: int = 3, forecast_length: int = 1,
                 backcast_length: int = 24, hidden_layer_units: int = 256,
                 epochs: int = 100, batch_size: int = 32, **kwargs):
        super().__init__(**kwargs)
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for N-BEATS. Install with: pip install torch")

        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for N-BEATS training."""
        X_seq, y_seq = [], []
        for i in range(self.backcast_length, len(X) - self.forecast_length + 1):
            X_seq.append(X[i-self.backcast_length:i].flatten())
            y_seq.append(y[i:i+self.forecast_length])
        return np.array(X_seq), np.array(y_seq)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NBeatsForecaster':
        """Fit N-BEATS model (simplified)."""
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)

        # Flatten sequences for N-BEATS
        X_flat = X_seq.reshape(X_seq.shape[0], -1)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_flat).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        input_size = X_flat.shape[1]
        self.model = NBeatsNet(input_size, self.forecast_length, self.hidden_layer_units).to(self.device)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Train model
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Create sequences for prediction
        if len(X) < self.backcast_length:
            X_padded = np.vstack([X[0:1]] * (self.backcast_length - len(X)) + [X])
        else:
            X_padded = X

        X_seq, _ = self._create_sequences(X_padded, np.zeros(len(X_padded)))

        # Flatten sequences
        X_flat = X_seq.reshape(X_seq.shape[0], -1)

        # Convert to tensor and predict
        X_tensor = torch.FloatTensor(X_flat).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()

        # Ensure predictions match expected length
        expected_length = len(X)
        if len(predictions) != expected_length:
            # Repeat last prediction or truncate to match expected length
            if len(predictions) < expected_length:
                predictions = np.pad(predictions, (0, expected_length - len(predictions)),
                                   mode='edge')
            else:
                predictions = predictions[:expected_length]

        return predictions

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        predictions = self.predict(X)
        uncertainties = np.full_like(predictions, np.std(predictions) * 0.1)
        return predictions, uncertainties


class BaselineForecasters:
    """Manager class for all baseline forecasting models."""

    @staticmethod
    def get_available_models() -> Dict[str, type]:
        """Get dictionary of available baseline models."""
        models = {
            'arima': ARIMAForecaster,
            'sarima': SARIMAForecaster,
            'svr': SVRForecaster,
            'mlp': MLPForecaster,
            'vanilla_rf': VanillaRandomForestForecaster,
        }

        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = LightGBMForecaster

        if PYTORCH_AVAILABLE:
            models.update({
                'lstm': LSTMForecaster,
                'cnn_lstm': CNNLSTMForecaster,
                'nbeats': NBeatsForecaster,
            })

        return models

    @staticmethod
    def create_model(model_name: str, **kwargs) -> BaseForecaster:
        """Create a baseline model by name."""
        available_models = BaselineForecasters.get_available_models()

        if model_name not in available_models:
            available_names = list(available_models.keys())
            raise ValueError(f"Unknown model: {model_name}. Available models: {available_names}")

        model_class = available_models[model_name]
        return model_class(**kwargs)

    @staticmethod
    def get_model_configs() -> Dict[str, Dict[str, Any]]:
        """Get default configurations for all baseline models."""
        configs = {
            'arima': {
                'order': (1, 1, 1)
            },
            'sarima': {
                'order': (1, 1, 1),
                'seasonal_order': (1, 1, 1, 24)
            },
            'svr': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale'
            },
            'mlp': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'max_iter': 1000
            },
            'vanilla_rf': {
                'n_estimators': 100,
                'max_depth': 20
            }
        }

        if LIGHTGBM_AVAILABLE:
            configs['lightgbm'] = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': -1
            }

        if PYTORCH_AVAILABLE:
            configs.update({
                'lstm': {
                    'lstm_units': 50,
                    'dense_units': 25,
                    'epochs': 100,
                    'batch_size': 32,
                    'sequence_length': 24
                },
                'cnn_lstm': {
                    'cnn_filters': 64,
                    'kernel_size': 3,
                    'lstm_units': 50,
                    'dense_units': 25,
                    'epochs': 100,
                    'batch_size': 32,
                    'sequence_length': 24
                },
                'nbeats': {
                    'stack_types': ['trend', 'seasonality'],
                    'nb_blocks_per_stack': 3,
                    'forecast_length': 1,
                    'backcast_length': 24,
                    'hidden_layer_units': 256,
                    'epochs': 100,
                    'batch_size': 32
                }
            })

        return configs

    @staticmethod
    def create_all_models(**common_kwargs) -> Dict[str, BaseForecaster]:
        """Create all available baseline models with default configurations."""
        models = {}
        available_models = BaselineForecasters.get_available_models()
        configs = BaselineForecasters.get_model_configs()

        for model_name in available_models:
            try:
                model_config = configs.get(model_name, {})
                model_config.update(common_kwargs)
                models[model_name] = BaselineForecasters.create_model(model_name, **model_config)
                logger.info(f"Created baseline model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to create model {model_name}: {e}")

        return models
