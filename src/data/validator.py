"""Data validation utilities for renewable energy forecasting."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataValidator:
    """Validator for renewable energy time series data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data validator.
        
        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config or {}
        self.min_data_points = self.config.get('min_data_points', 100)
        self.max_missing_ratio = self.config.get('max_missing_ratio', 0.1)
        self.max_outlier_ratio = self.config.get('max_outlier_ratio', 0.05)
        self.expected_frequency = self.config.get('expected_frequency', 'H')  # Hourly
        
    def validate_dataset(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive validation of the dataset.
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        logger.info("Starting comprehensive data validation")
        
        validation_report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'checks_performed': []
        }
        
        # Check 1: Basic structure validation
        structure_valid = self._validate_structure(data, validation_report)
        
        # Check 2: Data completeness
        completeness_valid = self._validate_completeness(data, validation_report)
        
        # Check 3: Data quality
        quality_valid = self._validate_quality(data, validation_report)
        
        # Check 4: Temporal consistency
        temporal_valid = self._validate_temporal_consistency(data, validation_report)
        
        # Check 5: Value ranges
        range_valid = self._validate_value_ranges(data, validation_report)
        
        # Overall validation result
        validation_report['is_valid'] = all([
            structure_valid, completeness_valid, quality_valid, 
            temporal_valid, range_valid
        ])
        
        # Generate summary statistics
        validation_report['statistics'] = self._generate_statistics(data)
        
        logger.info(f"Data validation completed. Valid: {validation_report['is_valid']}")
        if validation_report['errors']:
            logger.error(f"Validation errors: {validation_report['errors']}")
        if validation_report['warnings']:
            logger.warning(f"Validation warnings: {validation_report['warnings']}")
        
        return validation_report['is_valid'], validation_report
    
    def _validate_structure(self, data: pd.DataFrame, report: Dict[str, Any]) -> bool:
        """Validate basic data structure."""
        report['checks_performed'].append('structure')
        
        required_columns = ['timestamp', 'energy_generation']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            report['errors'].append(error_msg)
            return False
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            except Exception as e:
                error_msg = f"Cannot convert timestamp column to datetime: {e}"
                report['errors'].append(error_msg)
                return False
        
        if not pd.api.types.is_numeric_dtype(data['energy_generation']):
            try:
                data['energy_generation'] = pd.to_numeric(data['energy_generation'])
            except Exception as e:
                error_msg = f"Cannot convert energy_generation to numeric: {e}"
                report['errors'].append(error_msg)
                return False
        
        return True
    
    def _validate_completeness(self, data: pd.DataFrame, report: Dict[str, Any]) -> bool:
        """Validate data completeness."""
        report['checks_performed'].append('completeness')
        
        # Check minimum data points
        if len(data) < self.min_data_points:
            error_msg = f"Insufficient data points: {len(data)} < {self.min_data_points}"
            report['errors'].append(error_msg)
            return False
        
        # Check missing values
        missing_count = data['energy_generation'].isnull().sum()
        missing_ratio = missing_count / len(data)
        
        if missing_ratio > self.max_missing_ratio:
            error_msg = f"Too many missing values: {missing_ratio:.2%} > {self.max_missing_ratio:.2%}"
            report['errors'].append(error_msg)
            return False
        elif missing_ratio > 0:
            warning_msg = f"Found {missing_count} missing values ({missing_ratio:.2%})"
            report['warnings'].append(warning_msg)
        
        return True
    
    def _validate_quality(self, data: pd.DataFrame, report: Dict[str, Any]) -> bool:
        """Validate data quality."""
        report['checks_performed'].append('quality')
        
        values = data['energy_generation'].dropna().values
        
        # Check for outliers using IQR method
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = (values < lower_bound) | (values > upper_bound)
        outlier_ratio = np.sum(outliers) / len(values)
        
        if outlier_ratio > self.max_outlier_ratio:
            warning_msg = f"High outlier ratio: {outlier_ratio:.2%} > {self.max_outlier_ratio:.2%}"
            report['warnings'].append(warning_msg)
        
        # Check for negative values
        negative_count = np.sum(values < 0)
        if negative_count > 0:
            warning_msg = f"Found {negative_count} negative energy generation values"
            report['warnings'].append(warning_msg)
        
        # Check for constant values
        if np.std(values) == 0:
            error_msg = "Energy generation values are constant (zero variance)"
            report['errors'].append(error_msg)
            return False
        
        return True
    
    def _validate_temporal_consistency(self, data: pd.DataFrame, report: Dict[str, Any]) -> bool:
        """Validate temporal consistency."""
        report['checks_performed'].append('temporal_consistency')
        
        # Check if timestamps are sorted
        if not data['timestamp'].is_monotonic_increasing:
            error_msg = "Timestamps are not in ascending order"
            report['errors'].append(error_msg)
            return False
        
        # Check for duplicate timestamps
        duplicate_count = data['timestamp'].duplicated().sum()
        if duplicate_count > 0:
            error_msg = f"Found {duplicate_count} duplicate timestamps"
            report['errors'].append(error_msg)
            return False
        
        # Check frequency consistency
        if len(data) > 1:
            time_diffs = data['timestamp'].diff().dropna()
            
            # Expected frequency mapping
            freq_mapping = {
                'H': timedelta(hours=1),
                'D': timedelta(days=1),
                '15T': timedelta(minutes=15),
                '30T': timedelta(minutes=30)
            }
            
            expected_diff = freq_mapping.get(self.expected_frequency, timedelta(hours=1))
            
            # Check if most intervals match expected frequency
            correct_intervals = (time_diffs == expected_diff).sum()
            interval_consistency = correct_intervals / len(time_diffs)
            
            if interval_consistency < 0.9:  # 90% threshold
                warning_msg = f"Irregular time intervals: {interval_consistency:.2%} match expected frequency"
                report['warnings'].append(warning_msg)
        
        return True
    
    def _validate_value_ranges(self, data: pd.DataFrame, report: Dict[str, Any]) -> bool:
        """Validate value ranges."""
        report['checks_performed'].append('value_ranges')
        
        values = data['energy_generation'].dropna().values
        
        # Check for extremely large values (potential data errors)
        max_reasonable = self.config.get('max_reasonable_generation', 10000)  # MW
        large_values = np.sum(values > max_reasonable)
        
        if large_values > 0:
            warning_msg = f"Found {large_values} values > {max_reasonable} MW (potentially unrealistic)"
            report['warnings'].append(warning_msg)
        
        # Check value distribution
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Check for extremely high variance
        cv = std_val / mean_val if mean_val > 0 else float('inf')
        if cv > 2.0:  # Coefficient of variation > 200%
            warning_msg = f"High coefficient of variation: {cv:.2f} (data may be very noisy)"
            report['warnings'].append(warning_msg)
        
        return True
    
    def _generate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistics for the dataset."""
        values = data['energy_generation'].dropna().values
        
        stats = {
            'total_points': len(data),
            'valid_points': len(values),
            'missing_points': len(data) - len(values),
            'missing_ratio': (len(data) - len(values)) / len(data),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'zero_values': np.sum(values == 0),
            'negative_values': np.sum(values < 0)
        }
        
        # Time span statistics
        if len(data) > 0:
            time_span = data['timestamp'].max() - data['timestamp'].min()
            stats['time_span_days'] = time_span.total_seconds() / (24 * 3600)
            stats['start_date'] = data['timestamp'].min().isoformat()
            stats['end_date'] = data['timestamp'].max().isoformat()
        
        return stats
    
    def validate_forecast_inputs(self, data: np.ndarray, horizon: int) -> Tuple[bool, List[str]]:
        """Validate inputs for forecasting.
        
        Args:
            data: Input data array
            horizon: Forecast horizon
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check data array
        if not isinstance(data, np.ndarray):
            errors.append("Input data must be a numpy array")
        elif len(data) == 0:
            errors.append("Input data is empty")
        elif np.any(np.isnan(data)):
            errors.append("Input data contains NaN values")
        elif np.any(np.isinf(data)):
            errors.append("Input data contains infinite values")
        
        # Check horizon
        if not isinstance(horizon, int) or horizon <= 0:
            errors.append("Forecast horizon must be a positive integer")
        elif horizon > len(data):
            errors.append(f"Forecast horizon ({horizon}) cannot be larger than data length ({len(data)})")
        
        return len(errors) == 0, errors
    
    def validate_model_outputs(self, predictions: np.ndarray, 
                             intervals: Optional[np.ndarray] = None) -> Tuple[bool, List[str]]:
        """Validate model outputs.
        
        Args:
            predictions: Prediction array
            intervals: Optional prediction intervals
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check predictions
        if not isinstance(predictions, np.ndarray):
            errors.append("Predictions must be a numpy array")
        elif len(predictions) == 0:
            errors.append("Predictions array is empty")
        elif np.any(np.isnan(predictions)):
            errors.append("Predictions contain NaN values")
        elif np.any(np.isinf(predictions)):
            errors.append("Predictions contain infinite values")
        elif np.any(predictions < 0):
            errors.append("Predictions contain negative values")
        
        # Check intervals if provided
        if intervals is not None:
            if not isinstance(intervals, np.ndarray):
                errors.append("Intervals must be a numpy array")
            elif intervals.shape[1] != 2:
                errors.append("Intervals must have shape (n_samples, 2)")
            elif np.any(intervals[:, 0] > intervals[:, 1]):
                errors.append("Lower bounds must be <= upper bounds")
            elif np.any(intervals < 0):
                errors.append("Intervals contain negative values")
        
        return len(errors) == 0, errors