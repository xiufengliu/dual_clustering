"""Dual clustering implementation combining K-Means and FCM."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from .kmeans_clusterer import KMeansClusterer
from .fcm_clusterer import FCMClusterer

logger = logging.getLogger(__name__)


class DualClusterer:
    """Dual clustering approach combining K-Means and Fuzzy C-Means."""
    
    def __init__(self, n_clusters: int = 5, fcm_fuzziness: float = 2.0,
                 max_iter: int = 100, tol: float = 1e-4,
                 random_state: Optional[int] = None):
        """Initialize dual clusterer.

        Args:
            n_clusters: Number of clusters for both algorithms
            fcm_fuzziness: Fuzziness parameter for FCM
            max_iter: Maximum iterations for both algorithms
            tol: Convergence tolerance
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.fcm_fuzziness = fcm_fuzziness
        self.max_iter = max_iter
        # Ensure tol is always a float (in case it comes from YAML as string)
        self.tol = float(tol)
        self.random_state = random_state
        
        # Initialize clusterers
        self.kmeans_clusterer = KMeansClusterer(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=self.tol,  # Use the converted float value
            random_state=random_state
        )
        
        self.fcm_clusterer = FCMClusterer(
            n_clusters=n_clusters,
            m=fcm_fuzziness,
            max_iter=max_iter,
            tol=self.tol,  # Use the converted float value
            random_state=random_state
        )
        
        self.is_fitted = False
        self.integrated_features_ = None
        
    def fit(self, X: np.ndarray) -> 'DualClusterer':
        """Fit both K-Means and FCM to the data.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Self
        """
        logger.info(f"Fitting dual clustering on data shape {X.shape}")
        
        # Reshape if 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Fit K-Means
        logger.info("Fitting K-Means clusterer")
        self.kmeans_clusterer.fit(X)
        
        # Fit FCM
        logger.info("Fitting FCM clusterer")
        self.fcm_clusterer.fit(X)
        
        # Generate integrated features
        self.integrated_features_ = self._create_integrated_features()
        
        self.is_fitted = True
        logger.info("Dual clustering fitting completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict cluster assignments for new data.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Tuple of (kmeans_labels, fcm_memberships)
        """
        if not self.is_fitted:
            raise ValueError("DualClusterer must be fitted before prediction")
        
        # Reshape if 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Get predictions from both clusterers
        kmeans_labels = self.kmeans_clusterer.predict(X)
        fcm_memberships = self.fcm_clusterer.predict_proba(X)
        
        return kmeans_labels, fcm_memberships
    
    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit dual clustering and predict assignments.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Tuple of (kmeans_labels, fcm_memberships)
        """
        self.fit(X)
        return self.get_cluster_assignments()
    
    def get_cluster_assignments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get cluster assignments from fitted clusterers.
        
        Returns:
            Tuple of (kmeans_labels, fcm_memberships)
        """
        if not self.is_fitted:
            raise ValueError("DualClusterer must be fitted before accessing assignments")
        
        kmeans_labels = self.kmeans_clusterer.labels_
        fcm_memberships = self.fcm_clusterer.get_membership_matrix()
        
        return kmeans_labels, fcm_memberships
    
    def get_integrated_features(self) -> np.ndarray:
        """Get integrated cluster features as per Proposition 2 from the paper.
        
        Returns:
            Integrated feature matrix of shape (n_samples, 2*n_clusters)
            Format: [one_hot_kmeans, fcm_memberships]
        """
        if not self.is_fitted:
            raise ValueError("DualClusterer must be fitted before accessing integrated features")
        
        return self.integrated_features_
    
    def _create_integrated_features(self) -> np.ndarray:
        """Create integrated cluster features as described in the paper.
        
        Implementation of Proposition 2: f_i^cluster = [one_hot(k_i), u_i]
        
        Returns:
            Integrated feature matrix
        """
        kmeans_labels, fcm_memberships = self.get_cluster_assignments()
        n_samples = len(kmeans_labels)
        
        # Create one-hot encoding for K-Means labels
        one_hot_kmeans = np.zeros((n_samples, self.n_clusters))
        one_hot_kmeans[np.arange(n_samples), kmeans_labels] = 1
        
        # Concatenate one-hot K-Means with FCM memberships
        integrated_features = np.concatenate([one_hot_kmeans, fcm_memberships], axis=1)
        
        logger.info(f"Created integrated features with shape {integrated_features.shape}")
        
        return integrated_features
    
    def get_cluster_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get cluster centers from both clusterers.
        
        Returns:
            Tuple of (kmeans_centers, fcm_centers)
        """
        if not self.is_fitted:
            raise ValueError("DualClusterer must be fitted before accessing cluster centers")
        
        kmeans_centers = self.kmeans_clusterer.get_cluster_centers()
        fcm_centers = self.fcm_clusterer.get_cluster_centers()
        
        return kmeans_centers, fcm_centers
    
    def get_cluster_agreement(self) -> Dict[str, Any]:
        """Analyze agreement between K-Means and FCM clustering results.
        
        Returns:
            Dictionary with agreement metrics
        """
        if not self.is_fitted:
            raise ValueError("DualClusterer must be fitted before analyzing agreement")
        
        kmeans_labels, fcm_memberships = self.get_cluster_assignments()
        fcm_hard_labels = np.argmax(fcm_memberships, axis=1)
        
        # Calculate agreement metrics
        exact_agreement = np.mean(kmeans_labels == fcm_hard_labels)
        
        # Calculate average membership strength for K-Means assigned clusters
        n_samples = len(kmeans_labels)
        kmeans_membership_strength = np.zeros(n_samples)
        
        for i in range(n_samples):
            kmeans_cluster = kmeans_labels[i]
            kmeans_membership_strength[i] = fcm_memberships[i, kmeans_cluster]
        
        avg_membership_strength = np.mean(kmeans_membership_strength)
        
        # Calculate cluster center distances
        kmeans_centers, fcm_centers = self.get_cluster_centers()
        center_distances = np.zeros(self.n_clusters)
        
        for i in range(self.n_clusters):
            center_distances[i] = np.linalg.norm(kmeans_centers[i] - fcm_centers[i])
        
        agreement_metrics = {
            'exact_agreement_ratio': exact_agreement,
            'avg_membership_strength': avg_membership_strength,
            'center_distances': center_distances,
            'avg_center_distance': np.mean(center_distances),
            'max_center_distance': np.max(center_distances),
            'min_center_distance': np.min(center_distances)
        }
        
        return agreement_metrics
    
    def get_comprehensive_info(self) -> Dict[str, Any]:
        """Get comprehensive information about dual clustering results.
        
        Returns:
            Dictionary with detailed clustering information
        """
        if not self.is_fitted:
            raise ValueError("DualClusterer must be fitted before accessing comprehensive info")
        
        # Get individual clusterer info
        kmeans_info = self.kmeans_clusterer.get_cluster_info()
        fcm_info = self.fcm_clusterer.get_cluster_info()
        
        # Get agreement metrics
        agreement_info = self.get_cluster_agreement()
        
        # Combine all information
        comprehensive_info = {
            'dual_clustering_params': {
                'n_clusters': self.n_clusters,
                'fcm_fuzziness': self.fcm_fuzziness,
                'max_iter': self.max_iter,
                'tol': self.tol,
                'random_state': self.random_state
            },
            'kmeans_info': kmeans_info,
            'fcm_info': fcm_info,
            'agreement_metrics': agreement_info,
            'integrated_features_shape': self.integrated_features_.shape
        }
        
        return comprehensive_info
    
    def validate_clustering_quality(self) -> Dict[str, Any]:
        """Validate the quality of dual clustering results.
        
        Returns:
            Dictionary with quality metrics and validation results
        """
        if not self.is_fitted:
            raise ValueError("DualClusterer must be fitted before validation")
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'quality_metrics': {}
        }
        
        # Check cluster balance
        kmeans_labels, _ = self.get_cluster_assignments()
        unique, counts = np.unique(kmeans_labels, return_counts=True)
        
        min_cluster_size = np.min(counts)
        max_cluster_size = np.max(counts)
        cluster_balance_ratio = min_cluster_size / max_cluster_size
        
        validation_results['quality_metrics']['cluster_balance_ratio'] = cluster_balance_ratio
        
        if cluster_balance_ratio < 0.1:  # Very imbalanced clusters
            validation_results['warnings'].append(
                f"Highly imbalanced clusters detected (ratio: {cluster_balance_ratio:.3f})"
            )
        
        # Check agreement between clusterers
        agreement_metrics = self.get_cluster_agreement()
        exact_agreement = agreement_metrics['exact_agreement_ratio']
        
        validation_results['quality_metrics']['exact_agreement'] = exact_agreement
        
        if exact_agreement < 0.5:  # Low agreement
            validation_results['warnings'].append(
                f"Low agreement between K-Means and FCM ({exact_agreement:.3f})"
            )
        
        # Check FCM partition coefficient
        fcm_info = self.fcm_clusterer.get_cluster_info()
        partition_coefficient = fcm_info['partition_coefficient']
        
        validation_results['quality_metrics']['partition_coefficient'] = partition_coefficient
        
        if partition_coefficient < 0.7:  # Very fuzzy clustering
            validation_results['warnings'].append(
                f"Very fuzzy clustering detected (PC: {partition_coefficient:.3f})"
            )
        
        return validation_results