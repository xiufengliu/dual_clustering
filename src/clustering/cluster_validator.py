"""Cluster validation utilities."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging

logger = logging.getLogger(__name__)


class ClusterValidator:
    """Validates clustering results using various metrics."""
    
    def __init__(self):
        """Initialize cluster validator."""
        pass
    
    def validate_clustering(self, X: np.ndarray, labels: np.ndarray, 
                          centers: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Validate clustering results using multiple metrics.
        
        Args:
            X: Data points
            labels: Cluster labels
            centers: Cluster centers (optional)
            
        Returns:
            Dictionary with validation metrics
        """
        metrics = {}
        
        # Check if we have valid clustering
        n_clusters = len(np.unique(labels))
        if n_clusters < 2:
            logger.warning("Less than 2 clusters found, cannot compute validation metrics")
            return {"n_clusters": n_clusters, "valid": False}
        
        try:
            # Silhouette score
            metrics['silhouette_score'] = silhouette_score(X, labels)
            
            # Calinski-Harabasz score
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            
            # Davies-Bouldin score
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
            
            # Inertia (within-cluster sum of squares)
            metrics['inertia'] = self._calculate_inertia(X, labels, centers)
            
            # Number of clusters
            metrics['n_clusters'] = n_clusters
            
            # Cluster sizes
            unique_labels, counts = np.unique(labels, return_counts=True)
            metrics['cluster_sizes'] = dict(zip(unique_labels.tolist(), counts.tolist()))
            
            # Check for balanced clusters
            metrics['cluster_balance'] = self._calculate_cluster_balance(counts)
            
            metrics['valid'] = True
            
        except Exception as e:
            logger.error(f"Error computing validation metrics: {e}")
            metrics = {"n_clusters": n_clusters, "valid": False, "error": str(e)}
        
        return metrics
    
    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray, 
                          centers: Optional[np.ndarray] = None) -> float:
        """Calculate within-cluster sum of squares (inertia)."""
        if centers is None:
            # Calculate centers from data
            unique_labels = np.unique(labels)
            centers = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
        
        inertia = 0.0
        for i, label in enumerate(np.unique(labels)):
            cluster_points = X[labels == label]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centers[i]) ** 2)
        
        return inertia
    
    def _calculate_cluster_balance(self, cluster_sizes: np.ndarray) -> float:
        """Calculate cluster balance (1.0 = perfectly balanced, 0.0 = completely imbalanced)."""
        if len(cluster_sizes) <= 1:
            return 1.0
        
        # Calculate coefficient of variation
        mean_size = np.mean(cluster_sizes)
        std_size = np.std(cluster_sizes)
        
        if mean_size == 0:
            return 0.0
        
        cv = std_size / mean_size
        # Convert to balance score (lower CV = higher balance)
        balance = 1.0 / (1.0 + cv)
        
        return balance
    
    def recommend_n_clusters(self, X: np.ndarray, max_clusters: int = 10,
                           method: str = 'silhouette') -> Tuple[int, Dict[str, Any]]:
        """Recommend optimal number of clusters.
        
        Args:
            X: Data points
            max_clusters: Maximum number of clusters to test
            method: Method to use ('silhouette', 'elbow', 'gap')
            
        Returns:
            Tuple of (recommended_n_clusters, validation_results)
        """
        from sklearn.cluster import KMeans
        
        results = {}
        scores = []
        
        for n_clusters in range(2, min(max_clusters + 1, len(X))):
            try:
                # Fit K-means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                
                # Calculate validation metrics
                validation = self.validate_clustering(X, labels, kmeans.cluster_centers_)
                
                if validation.get('valid', False):
                    if method == 'silhouette':
                        score = validation['silhouette_score']
                    elif method == 'elbow':
                        score = -validation['inertia']  # Negative because we want to minimize
                    else:
                        score = validation['silhouette_score']  # Default to silhouette
                    
                    scores.append(score)
                    results[n_clusters] = validation
                else:
                    scores.append(-np.inf)
                    results[n_clusters] = validation
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate {n_clusters} clusters: {e}")
                scores.append(-np.inf)
                results[n_clusters] = {"valid": False, "error": str(e)}
        
        if not scores or all(s == -np.inf for s in scores):
            logger.warning("Could not find valid clustering for any number of clusters")
            return 2, results
        
        # Find optimal number of clusters
        if method == 'elbow':
            # Use elbow method (look for point of maximum curvature)
            optimal_idx = self._find_elbow_point(scores)
            optimal_n_clusters = optimal_idx + 2  # +2 because we start from 2 clusters
        else:
            # Use maximum score
            optimal_idx = np.argmax(scores)
            optimal_n_clusters = optimal_idx + 2  # +2 because we start from 2 clusters
        
        return optimal_n_clusters, results
    
    def _find_elbow_point(self, scores: list) -> int:
        """Find elbow point in scores using the kneedle algorithm (simplified)."""
        if len(scores) < 3:
            return 0
        
        # Calculate second derivatives
        second_derivatives = []
        for i in range(1, len(scores) - 1):
            second_deriv = scores[i-1] - 2*scores[i] + scores[i+1]
            second_derivatives.append(abs(second_deriv))
        
        if not second_derivatives:
            return 0
        
        # Find point of maximum curvature
        elbow_idx = np.argmax(second_derivatives) + 1  # +1 to account for offset
        return elbow_idx
    
    def compare_clusterings(self, X: np.ndarray, labels1: np.ndarray, 
                          labels2: np.ndarray) -> Dict[str, float]:
        """Compare two clustering results.
        
        Args:
            X: Data points
            labels1: First clustering labels
            labels2: Second clustering labels
            
        Returns:
            Dictionary with comparison metrics
        """
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        comparison = {}
        
        try:
            # Adjusted Rand Index
            comparison['adjusted_rand_score'] = adjusted_rand_score(labels1, labels2)
            
            # Normalized Mutual Information
            comparison['normalized_mutual_info'] = normalized_mutual_info_score(labels1, labels2)
            
            # Individual validation metrics
            validation1 = self.validate_clustering(X, labels1)
            validation2 = self.validate_clustering(X, labels2)
            
            comparison['clustering1_metrics'] = validation1
            comparison['clustering2_metrics'] = validation2
            
            # Determine which is better based on silhouette score
            if (validation1.get('valid', False) and validation2.get('valid', False)):
                sil1 = validation1.get('silhouette_score', -1)
                sil2 = validation2.get('silhouette_score', -1)
                comparison['better_clustering'] = 1 if sil1 > sil2 else 2
            else:
                comparison['better_clustering'] = None
            
        except Exception as e:
            logger.error(f"Error comparing clusterings: {e}")
            comparison['error'] = str(e)
        
        return comparison
