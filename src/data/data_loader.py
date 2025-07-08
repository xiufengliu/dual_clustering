"""Data loading utilities for renewable energy datasets."""

import pandas as pd
import numpy as np
import requests
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load_data(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """Load data for the specified date range."""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate loaded data."""
        pass


class ENTSOEDataLoader(BaseDataLoader):
    """Data loader for ENTSO-E Transparency Platform data."""
    
    def __init__(self, api_token: Optional[str] = None, cache_dir: str = "data/raw"):
        """Initialize ENTSO-E data loader.
        
        Args:
            api_token: ENTSO-E API token (if None, will try to load from cache)
            cache_dir: Directory for caching downloaded data
        """
        self.api_token = api_token
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # ENTSO-E area codes
        self.area_codes = {
            "Denmark": "10YDK-1--------W",
            "Germany": "10Y1001A1001A83F",
            "Netherlands": "10YNL----------L",
            "Belgium": "10YBE----------2"
        }
        
        # Process type codes
        self.process_types = {
            "solar": "A75",  # Solar generation
            "wind_onshore": "A73",  # Wind onshore generation
            "wind_offshore": "A74",  # Wind offshore generation
            "total_load": "A65"  # Total load
        }
    
    def load_data(self, start_date: str, end_date: str, 
                  country: str = "Denmark", energy_type: str = "solar",
                  use_cache: bool = True) -> pd.DataFrame:
        """Load renewable energy data from ENTSO-E or cache.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            country: Country name
            energy_type: Type of energy (solar, wind_onshore, wind_offshore)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with timestamp and energy generation columns
        """
        cache_file = self._get_cache_filename(start_date, end_date, country, energy_type)
        
        # Try to load from cache first
        if use_cache and cache_file.exists():
            logger.info(f"Loading data from cache: {cache_file}")
            try:
                data = pd.read_csv(cache_file, parse_dates=['timestamp'])
                if self.validate_data(data):
                    return data
                else:
                    logger.warning("Cached data validation failed, downloading fresh data")
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
        
        # Download fresh data
        if self.api_token:
            data = self._download_from_api(start_date, end_date, country, energy_type)
        else:
            # Generate synthetic data for demonstration
            logger.warning("No API token provided, generating synthetic data")
            data = self._generate_synthetic_data(start_date, end_date, energy_type)
        
        # Cache the data
        if use_cache:
            data.to_csv(cache_file, index=False)
            logger.info(f"Data cached to: {cache_file}")
        
        return data
    
    def load_solar_data(self, start_date: str, end_date: str, 
                       country: str = "Denmark") -> pd.DataFrame:
        """Load solar power generation data."""
        return self.load_data(start_date, end_date, country, "solar")
    
    def load_wind_data(self, start_date: str, end_date: str, 
                      country: str = "Denmark", offshore: bool = False) -> pd.DataFrame:
        """Load wind power generation data."""
        energy_type = "wind_offshore" if offshore else "wind_onshore"
        return self.load_data(start_date, end_date, country, energy_type)
    
    def _download_from_api(self, start_date: str, end_date: str, 
                          country: str, energy_type: str) -> pd.DataFrame:
        """Download data from ENTSO-E API."""
        base_url = "https://web-api.tp.entsoe.eu/api"
        
        params = {
            "securityToken": self.api_token,
            "documentType": "A75",  # Generation forecast
            "processType": self.process_types[energy_type],
            "in_Domain": self.area_codes[country],
            "periodStart": self._format_date_for_api(start_date),
            "periodEnd": self._format_date_for_api(end_date)
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response (simplified - would need proper XML parsing)
            # For now, return synthetic data
            logger.warning("API response parsing not implemented, generating synthetic data")
            return self._generate_synthetic_data(start_date, end_date, energy_type)
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            logger.info("Falling back to synthetic data generation")
            return self._generate_synthetic_data(start_date, end_date, energy_type)
    
    def _generate_synthetic_data(self, start_date: str, end_date: str, 
                                energy_type: str) -> pd.DataFrame:
        """Generate synthetic renewable energy data for testing."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Generate hourly timestamps
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq='H')
        n_points = len(timestamps)
        
        # Generate synthetic data based on energy type
        if energy_type == "solar":
            # Solar pattern: daily cycle with seasonal variation
            hours = np.array([ts.hour for ts in timestamps])
            days = np.array([ts.dayofyear for ts in timestamps])
            
            # Daily pattern (higher during day)
            daily_pattern = np.maximum(0, np.sin((hours - 6) * np.pi / 12))
            
            # Seasonal pattern (higher in summer)
            seasonal_pattern = 0.5 + 0.5 * np.sin((days - 80) * 2 * np.pi / 365)
            
            # Base generation with noise
            base_generation = 100 * daily_pattern * seasonal_pattern
            noise = np.random.normal(0, 10, n_points)
            generation = np.maximum(0, base_generation + noise)
            
        elif energy_type in ["wind_onshore", "wind_offshore"]:
            # Wind pattern: more variable, less predictable
            # Use AR(1) process with seasonal component
            base_level = 80 if energy_type == "wind_offshore" else 60
            
            generation = np.zeros(n_points)
            generation[0] = base_level
            
            for i in range(1, n_points):
                # AR(1) component
                ar_component = 0.7 * generation[i-1]
                
                # Seasonal component
                day_of_year = timestamps[i].dayofyear
                seasonal = 20 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
                
                # Random shock
                shock = np.random.normal(0, 15)
                
                generation[i] = max(0, 0.3 * base_level + 0.7 * ar_component + seasonal + shock)
        
        else:
            # Default pattern
            generation = np.random.uniform(20, 100, n_points)
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'energy_generation': generation
        })
        
        logger.info(f"Generated synthetic {energy_type} data: {len(data)} points from {start_date} to {end_date}")
        return data
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate loaded data."""
        required_columns = ['timestamp', 'energy_generation']
        
        # Check required columns
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns. Expected: {required_columns}, Got: {list(data.columns)}")
            return False
        
        # Check for empty data
        if len(data) == 0:
            logger.error("Data is empty")
            return False
        
        # Check for null values
        if data['energy_generation'].isnull().any():
            logger.warning("Data contains null values in energy_generation column")
        
        # Check for negative values
        if (data['energy_generation'] < 0).any():
            logger.warning("Data contains negative energy generation values")
        
        # Check timestamp ordering
        if not data['timestamp'].is_monotonic_increasing:
            logger.warning("Timestamps are not in ascending order")
        
        logger.info(f"Data validation completed. Shape: {data.shape}")
        return True
    
    def _get_cache_filename(self, start_date: str, end_date: str, 
                           country: str, energy_type: str) -> Path:
        """Generate cache filename for the data."""
        filename = f"{country}_{energy_type}_{start_date}_{end_date}.csv"
        return self.cache_dir / filename
    
    def _format_date_for_api(self, date_str: str) -> str:
        """Format date for ENTSO-E API."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y%m%d%H%M")


class CSVDataLoader(BaseDataLoader):
    """Data loader for CSV files."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize CSV data loader.
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
    
    def load_data(self, filename: str, timestamp_col: str = "timestamp",
                  value_col: str = "energy_generation", **kwargs) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            filename: CSV filename
            timestamp_col: Name of timestamp column
            value_col: Name of value column
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with standardized column names
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data
        data = pd.read_csv(file_path, **kwargs)
        
        # Standardize column names
        if timestamp_col in data.columns:
            data['timestamp'] = pd.to_datetime(data[timestamp_col])
        
        if value_col in data.columns:
            data['energy_generation'] = data[value_col]
        
        # Keep only required columns
        data = data[['timestamp', 'energy_generation']].copy()
        
        if not self.validate_data(data):
            raise ValueError("Data validation failed")
        
        logger.info(f"Loaded data from {file_path}: {len(data)} points")
        return data
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate loaded data."""
        required_columns = ['timestamp', 'energy_generation']
        
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns: {required_columns}")
            return False
        
        if len(data) == 0:
            logger.error("Data is empty")
            return False
        
        return True