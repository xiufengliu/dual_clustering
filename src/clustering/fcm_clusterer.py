"""Fuzzy C-Means clustering implementation."""

import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

from .base_clusterer import BaseClusterer
from ..utils.math_utils import safe_divide

logger = logging.getLogger(__name__)


class FCMClusterer(BaseClusterer):
    """Fuzzy C-Means clustering implementation."""
    
    def __init__(self, n_clusters: int = 5, m: float = 2.0, max_iter: int = 100,
                 tol: float = 1e-4, random_state: Optional[int] = None):
        """Initialize FCM clusterer.
        
        Args:
            n_clusters: Number of clusters
            m: Fuzziness parameter (> 1.0)
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            random_state: Random state for reproducibility
        """
        super().__init__(n_clusters, random_state)
        
        if m <= 1.0:
            raise ValueError("Fuzziness parameter m must be > 1.0")
        
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        
        # FCM-specific attributes
        self.membership_matrix_ = None
        self.objective_function_history_ = []
        self.n_iter_ = 0
        
    def fit(self, X: np.ndarray) -> 'FCMClusterer':
        """Fit FCM to the data.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Self
        """
        self.validate_input(X)
        
        logger.info(f"Fitting FCM with {self.n_clusters} clusters, m={self.m} on data shape {X.shape}")
        
        # Reshape if 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Initialize membership matrix randomly
        self.membership_matrix_ = self._initialize_membership_matrix(n_samples)
        
        # Initialize objective function history
        self.objective_function_history_ = []
        
        # Main FCM iteration loop
        for iteration in range(self.max_iter):
            # Update cluster centers
            self.cluster_centers_ = self._update_cluster_centers(X, self.membership_matrix_)
            
            # Update membership matrix
            new_membership_matrix = self._update_membership_matrix(X, self.cluster_centers_)
            
            # Calculate objective function
            objective_value = self._calculate_objective_function(X, self.cluster_centers_, new_membership_matrix)
            self.objective_function_history_.append(objective_value)
            
            # Check for convergence
            membership_change = np.linalg.norm(new_membership_matrix - self.membership_matrix_)
            
            if membership_change < self.tol:
                logger.info(f"FCM converged after {iteration + 1} iterations")
                break
            
            self.membership_matrix_ = new_membership_matrix
            
        else:
            logger.warning(f"FCM did not converge after {self.max_iter} iterations")
        
        self.n_iter_ = iteration + 1
        self.is_fitted = True
        
        logger.info(f"FCM fitting completed. Final objective: {self.objective_function_history_[-1]:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new data.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Hard cluster assignments (most likely cluster for each point)
        """
        membership_matrix = self.predict_proba(X)
        return np.argmax(membership_matrix, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict membership probabilities for new data.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Membership matrix of shape (n_samples, n_clusters)
        """
        if not self.is_fitted:
            raise ValueError("FCMClusterer must be fitted before prediction")
        
        self.validate_input(X)
        
        # Reshape if 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self._update_membership_matrix(X, self.cluster_centers_)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit FCM and predict cluster assignments.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Hard cluster assignments
        """
        self.fit(X)
        return np.argmax(self.membership_matrix_, axis=1)
    
    def _initialize_membership_matrix(self, n_samples: int) -> np.ndarray:
        """Initialize membership matrix randomly.
        
        Args:
            n_samples: Number of data samples
            
        Returns:
            Initial membership matrix of shape (n_samples, n_clusters)
        """
        # Random initialization
        membership_matrix = np.random.rand(n_samples, self.n_clusters)
        
        # Normalize so each row sums to 1
        row_sums = np.sum(membership_matrix, axis=1, keepdims=True)
        membership_matrix = safe_divide(membership_matrix, row_sums, default_value=1.0/self.n_clusters)
        
        return membership_matrix
    
    def _update_cluster_centers(self, X: np.ndarray, membership_matrix: np.ndarray) -> np.ndarray:
        """Update cluster centers based on current membership matrix.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            membership_matrix: Membership matrix of shape (n_samples, n_clusters)
            
        Returns:
            Updated cluster centers of shape (n_clusters, n_features)
        """
        # Raise membership values to power m
        membership_powered = np.power(membership_matrix, self.m)
        
        # Calculate weighted sums
        numerator = np.dot(membership_powered.T, X)
        denominator = np.sum(membership_powered, axis=0, keepdims=True).T
        
        # Compute centers with safe division
        centers = safe_divide(numerator, denominator, default_value=0.0)
        
        return centers
    
    def _update_membership_matrix(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Update membership matrix based on current cluster centers.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            centers: Cluster centers of shape (n_clusters, n_features)
            
        Returns:
            Updated membership matrix of shape (n_samples, n_clusters)
        """
        n_samples = X.shape[0]
        membership_matrix = np.zeros((n_samples, self.n_clusters))
        
        # Calculate distances from each point to each center
        for i, point in enumerate(X):
            distances = np.linalg.norm(point - centers, axis=1)
            
            # Handle case where point coincides with a center
            if np.any(distances == 0):
                zero_indices = np.where(distances == 0)[0]
                membership_matrix[i, zero_indices[0]] = 1.0
            else:
                # Standard FCM membership calculation
                for j in range(self.n_clusters):
                    sum_term = 0.0
                    for k in range(self.n_clusters):
                        ratio = distances[j] / distances[k]
                        sum_term += np.power(ratio, 2.0 / (self.m - 1.0))
                    
                    membership_matrix[i, j] = 1.0 / sum_term
        
        return membership_matrix
    
    def _calculate_objective_function(self, X: np.ndarray, centers: np.ndarray, 
                                    membership_matrix: np.ndarray) -> float:
        """Calculate FCM objective function value.
        
        Args:
            X: Data matrix
            centers: Cluster centers
            membership_matrix: Membership matrix
            
        Returns:
            Objective function value
        """
        objective = 0.0
        
        for i, point in enumerate(X):
            for j, center in enumerate(centers):
                distance_squared = np.sum((point - center) ** 2)
                membership_powered = np.power(membership_matrix[i, j], self.m)
                objective += membership_powered * distance_squared
        
        return objective
    
    def get_membership_matrix(self) -> np.ndarray:
        """Get the membership matrix.
        
        Returns:
            Membership matrix of shape (n_samples, n_clusters)
        """
        if not self.is_fitted:
            raise ValueError("FCMClusterer must be fitted before accessing membership matrix")
        
        return self.membership_matrix_
    
    def get_hard_clusters(self) -> np.ndarray:
        """Get hard cluster assignments (most likely cluster for each point).
        
        Returns:
            Hard cluster assignments
        """
        if not self.is_fitted:
            raise ValueError("FCMClusterer must be fitted before accessing hard clusters")
        
        return np.argmax(self.membership_matrix_, axis=1)
    
    def get_cluster_entropies(self) -> np.ndarray:
        """Calculate entropy for each data point based on membership distribution.
        
        Returns:
            Entropy values for each data point
        """
        if not self.is_fitted:
            raise ValueError("FCMClusterer must be fitted before calculating entropies")
        
        from ..utils.math_utils import compute_shannon_entropy
        
        entropies = np.zeros(self.membership_matrix_.shape[0])
        
        for i, membership_row in enumerate(self.membership_matrix_):
            entropies[i] = compute_shannon_entropy(membership_row)
        
        return entropies
    
    def get_params(self) -> Dict[str, Any]:
        """Get clusterer parameters.
        
        Returns:
            Dictionary of parameters
        """
        params = super().get_params()
        params.update({
            'm': self.m,
            'max_iter': self.max_iter,
            'tol': self.tol
        })
        return params
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get comprehensive cluster information.
        
        Returns:
            Dictionary with cluster information
        """
        if not self.is_fitted:
            raise ValueError("FCMClusterer must be fitted before accessing cluster info")
        
        hard_clusters = self.get_hard_clusters()
        unique, counts = np.unique(hard_clusters, return_counts=True)
        cluster_sizes = np.zeros(self.n_clusters)
        cluster_sizes[unique] = counts
        
        # Calculate average membership strength for each cluster
        avg_membership = np.mean(self.membership_matrix_, axis=0)
        
        # Calculate partition coefficient (measure of fuzziness)
        partition_coefficient = np.sum(self.membership_matrix_ ** 2) / self.membership_matrix_.shape[0]
        
        info = {
            'n_clusters': self.n_clusters,
            'cluster_centers': self.cluster_centers_,
            'membership_matrix': self.membership_matrix_,
            'hard_cluster_sizes': cluster_sizes,
            'avg_membership_strength': avg_membership,
            'partition_coefficient': partition_coefficient,
            'fuzziness_parameter': self.m,
            'n_iterations': self.n_iter_,
            'final_objective': self.objective_function_history_[-1] if self.objective_function_history_ else None,
            'objective_history': self.objective_function_history_
        }
        
        return info