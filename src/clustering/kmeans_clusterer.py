"""K-Means clustering implementation."""

import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, Dict, Any
import logging

from .base_clusterer import BaseClusterer

logger = logging.getLogger(__name__)


class KMeansClusterer(BaseClusterer):
    """K-Means clustering wrapper for the framework."""
    
    def __init__(self, n_clusters: int = 5, max_iter: int = 300, 
                 tol: float = 1e-4, random_state: Optional[int] = None,
                 init: str = 'k-means++', n_init: int = 10):
        """Initialize K-Means clusterer.
        
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            random_state: Random state for reproducibility
            init: Initialization method
            n_init: Number of random initializations
        """
        super().__init__(n_clusters, random_state)
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        
        # Initialize sklearn KMeans
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            init=self.init,
            n_init=self.n_init
        )
        
        self.labels_ = None
        self.inertia_ = None
    
    def fit(self, X: np.ndarray) -> 'KMeansClusterer':
        """Fit K-Means to the data.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Self
        """
        self.validate_input(X)
        
        logger.info(f"Fitting K-Means with {self.n_clusters} clusters on data shape {X.shape}")
        
        # Reshape if 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Fit the model
        self.kmeans.fit(X)
        
        # Store results
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.labels_ = self.kmeans.labels_
        self.inertia_ = self.kmeans.inertia_
        self.is_fitted = True
        
        logger.info(f"K-Means fitting completed. Inertia: {self.inertia_:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new data.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Cluster assignments array of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("KMeansClusterer must be fitted before prediction")
        
        self.validate_input(X)
        
        # Reshape if 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self.kmeans.predict(X)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit K-Means and predict cluster assignments.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Cluster assignments array of shape (n_samples,)
        """
        self.fit(X)
        return self.labels_
    
    def get_cluster_distances(self, X: np.ndarray) -> np.ndarray:
        """Get distances from each point to all cluster centers.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Distance matrix of shape (n_samples, n_clusters)
        """
        if not self.is_fitted:
            raise ValueError("KMeansClusterer must be fitted before computing distances")
        
        self.validate_input(X)
        
        # Reshape if 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Compute distances to all centers
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i, center in enumerate(self.cluster_centers_):
            distances[:, i] = np.linalg.norm(X - center, axis=1)
        
        return distances
    
    def get_within_cluster_sum_of_squares(self) -> float:
        """Get within-cluster sum of squares (inertia).
        
        Returns:
            Within-cluster sum of squares
        """
        if not self.is_fitted:
            raise ValueError("KMeansClusterer must be fitted before accessing inertia")
        
        return self.inertia_
    
    def get_cluster_sizes(self) -> np.ndarray:
        """Get the size of each cluster.
        
        Returns:
            Array of cluster sizes
        """
        if not self.is_fitted:
            raise ValueError("KMeansClusterer must be fitted before accessing cluster sizes")
        
        unique, counts = np.unique(self.labels_, return_counts=True)
        cluster_sizes = np.zeros(self.n_clusters)
        cluster_sizes[unique] = counts
        
        return cluster_sizes
    
    def get_params(self) -> Dict[str, Any]:
        """Get clusterer parameters.
        
        Returns:
            Dictionary of parameters
        """
        params = super().get_params()
        params.update({
            'max_iter': self.max_iter,
            'tol': self.tol,
            'init': self.init,
            'n_init': self.n_init
        })
        return params
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get comprehensive cluster information.
        
        Returns:
            Dictionary with cluster information
        """
        if not self.is_fitted:
            raise ValueError("KMeansClusterer must be fitted before accessing cluster info")
        
        cluster_sizes = self.get_cluster_sizes()
        
        info = {
            'n_clusters': self.n_clusters,
            'cluster_centers': self.cluster_centers_,
            'cluster_sizes': cluster_sizes,
            'inertia': self.inertia_,
            'labels': self.labels_,
            'largest_cluster': np.argmax(cluster_sizes),
            'smallest_cluster': np.argmin(cluster_sizes),
            'cluster_size_ratio': np.max(cluster_sizes) / np.min(cluster_sizes) if np.min(cluster_sizes) > 0 else float('inf')
        }
        
        return info