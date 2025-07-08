"""Base class for clustering algorithms."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseClusterer(ABC):
    """Abstract base class for clustering algorithms."""
    
    def __init__(self, n_clusters: int, random_state: Optional[int] = None):
        """Initialize base clusterer.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.is_fitted = False
        self.cluster_centers_ = None
        
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClusterer':
        """Fit the clustering algorithm to data.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new data.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Cluster assignments
        """
        pass
    
    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the algorithm and predict cluster assignments.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Cluster assignments
        """
        pass
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers.
        
        Returns:
            Cluster centers array
        """
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted before accessing cluster centers")
        return self.cluster_centers_
    
    def validate_input(self, X: np.ndarray) -> None:
        """Validate input data.
        
        Args:
            X: Input data array
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        if X.ndim != 2:
            raise ValueError(f"Input must be 2D array, got {X.ndim}D")
        
        if len(X) == 0:
            raise ValueError("Input array is empty")
        
        if np.any(np.isnan(X)):
            raise ValueError("Input contains NaN values")
        
        if np.any(np.isinf(X)):
            raise ValueError("Input contains infinite values")
    
    def get_params(self) -> Dict[str, Any]:
        """Get clusterer parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'n_clusters': self.n_clusters,
            'random_state': self.random_state
        }
    
    def set_params(self, **params) -> 'BaseClusterer':
        """Set clusterer parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self