"""Data preprocessing utilities for renewable energy forecasting."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

from ..utils.math_utils import normalize_data, denormalize_data

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocessor for renewable energy time series data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or {}
        self.normalization_params = None
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.interpolation_method = self.config.get('interpolation_method', 'linear')
        self.normalization_method = self.config.get('normalization', 'min_max')
        
    def preprocess(self, data: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Complete preprocessing pipeline.
        
        Args:
            data: Input DataFrame with 'timestamp' and 'energy_generation' columns
            fit: Whether to fit preprocessing parameters (True for training data)
            
        Returns:
            Tuple of (processed_data, preprocessing_params)
        """
        logger.info("Starting data preprocessing pipeline")
        
        # Make a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Step 1: Handle missing values
        processed_data = self._handle_missing_values(processed_data)
        
        # Step 2: Handle outliers
        processed_data = self._handle_outliers(processed_data, fit=fit)
        
        # Step 3: Extract time series values
        values = processed_data['energy_generation'].values
        
        # Step 4: Normalize data
        if fit:
            normalized_values, self.normalization_params = normalize_data(
                values, method=self.normalization_method
            )
        else:
            if self.normalization_params is None:
                raise ValueError("Normalization parameters not fitted. Call preprocess with fit=True first.")
            normalized_values = denormalize_data(values, self.normalization_params)
        
        # Prepare preprocessing parameters
        preprocessing_params = {
            'normalization_params': self.normalization_params,
            'outlier_threshold': self.outlier_threshold,
            'interpolation_method': self.interpolation_method,
            'original_length': len(data),
            'processed_length': len(normalized_values)
        }
        
        logger.info(f"Preprocessing completed. Original length: {len(data)}, "
                   f"Processed length: {len(normalized_values)}")
        
        return normalized_values, preprocessing_params
    
    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """Transform normalized data back to original scale.
        
        Args:
            normalized_data: Normalized data array
            
        Returns:
            Data in original scale
        """
        if self.normalization_params is None:
            raise ValueError("Normalization parameters not available. Fit preprocessor first.")
        
        return denormalize_data(normalized_data, self.normalization_params)
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        missing_count = data['energy_generation'].isnull().sum()
        
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values, applying {self.interpolation_method} interpolation")
            
            if self.interpolation_method == 'linear':
                data['energy_generation'] = data['energy_generation'].interpolate(method='linear')
            elif self.interpolation_method == 'forward_fill':
                data['energy_generation'] = data['energy_generation'].fillna(method='ffill')
            elif self.interpolation_method == 'backward_fill':
                data['energy_generation'] = data['energy_generation'].fillna(method='bfill')
            elif self.interpolation_method == 'mean':
                mean_value = data['energy_generation'].mean()
                data['energy_generation'] = data['energy_generation'].fillna(mean_value)
            else:
                logger.warning(f"Unknown interpolation method: {self.interpolation_method}, using linear")
                data['energy_generation'] = data['energy_generation'].interpolate(method='linear')
            
            # Handle any remaining missing values at the edges
            data['energy_generation'] = data['energy_generation'].fillna(method='ffill').fillna(method='bfill')
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle outliers in the data.
        
        Args:
            data: Input DataFrame
            fit: Whether to compute outlier statistics
            
        Returns:
            DataFrame with outliers handled
        """
        values = data['energy_generation'].values
        
        if fit:
            # Compute outlier statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Store for later use
            self.outlier_stats = {'mean': mean_val, 'std': std_val}
        else:
            if not hasattr(self, 'outlier_stats'):
                logger.warning("Outlier statistics not available, skipping outlier handling")
                return data
            mean_val = self.outlier_stats['mean']
            std_val = self.outlier_stats['std']
        
        # Identify outliers using z-score
        z_scores = np.abs((values - mean_val) / std_val)
        outlier_mask = z_scores > self.outlier_threshold
        
        outlier_count = np.sum(outlier_mask)
        
        if outlier_count > 0:
            logger.info(f"Found {outlier_count} outliers (z-score > {self.outlier_threshold})")
            
            # Replace outliers with interpolated values
            data_copy = data.copy()
            data_copy.loc[outlier_mask, 'energy_generation'] = np.nan
            data_copy['energy_generation'] = data_copy['energy_generation'].interpolate(method='linear')
            
            # Handle edge cases
            data_copy['energy_generation'] = data_copy['energy_generation'].fillna(method='ffill').fillna(method='bfill')
            
            return data_copy
        
        return data
    
    def get_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute descriptive statistics for the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with data statistics
        """
        values = data['energy_generation'].values
        
        stats_dict = {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values),
            'missing_values': data['energy_generation'].isnull().sum(),
            'zero_values': np.sum(values == 0),
            'negative_values': np.sum(values < 0)
        }
        
        return stats_dict
    
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamps.
        
        Args:
            data: Input DataFrame with timestamp column
            
        Returns:
            DataFrame with additional time features
        """
        data_copy = data.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data_copy['timestamp']):
            data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
        
        # Extract time features
        data_copy['hour'] = data_copy['timestamp'].dt.hour
        data_copy['day_of_week'] = data_copy['timestamp'].dt.dayofweek
        data_copy['day_of_year'] = data_copy['timestamp'].dt.dayofyear
        data_copy['month'] = data_copy['timestamp'].dt.month
        data_copy['quarter'] = data_copy['timestamp'].dt.quarter
        data_copy['year'] = data_copy['timestamp'].dt.year
        
        # Cyclical encoding for periodic features
        data_copy['hour_sin'] = np.sin(2 * np.pi * data_copy['hour'] / 24)
        data_copy['hour_cos'] = np.cos(2 * np.pi * data_copy['hour'] / 24)
        
        data_copy['day_of_week_sin'] = np.sin(2 * np.pi * data_copy['day_of_week'] / 7)
        data_copy['day_of_week_cos'] = np.cos(2 * np.pi * data_copy['day_of_week'] / 7)
        
        data_copy['day_of_year_sin'] = np.sin(2 * np.pi * data_copy['day_of_year'] / 365)
        data_copy['day_of_year_cos'] = np.cos(2 * np.pi * data_copy['day_of_year'] / 365)
        
        logger.info("Created time-based features")
        return data_copy
    
    def create_lag_features(self, data: pd.DataFrame, lags: list = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """Create lagged features for time series.
        
        Args:
            data: Input DataFrame
            lags: List of lag values to create
            
        Returns:
            DataFrame with lag features
        """
        data_copy = data.copy()
        
        for lag in lags:
            data_copy[f'lag_{lag}'] = data_copy['energy_generation'].shift(lag)
        
        # Drop rows with NaN values created by lagging
        data_copy = data_copy.dropna()
        
        logger.info(f"Created lag features for lags: {lags}")
        return data_copy
    
    def split_data(self, data: pd.DataFrame, train_ratio: float = 0.8, 
                   val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets.
        
        Args:
            data: Input DataFrame
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        n_total = len(data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = data.iloc[:n_train].copy()
        val_data = data.iloc[n_train:n_train + n_val].copy()
        test_data = data.iloc[n_train + n_val:].copy()
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data