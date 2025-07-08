"""Neutrosophic transformation implementation based on the paper's methodology."""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ..utils.math_utils import compute_shannon_entropy, normalize_entropy

logger = logging.getLogger(__name__)


@dataclass
class NeutrosophicComponents:
    """Container for neutrosophic components (T, I, F)."""
    truth: np.ndarray
    indeterminacy: np.ndarray
    falsity: np.ndarray
    
    def __post_init__(self):
        """Validate neutrosophic components after initialization."""
        if not (self.truth.shape == self.indeterminacy.shape == self.falsity.shape):
            raise ValueError("All neutrosophic components must have the same shape")
        
        # Check value ranges
        for component, name in [(self.truth, "Truth"), (self.indeterminacy, "Indeterminacy"), (self.falsity, "Falsity")]:
            if np.any(component < 0) or np.any(component > 1):
                raise ValueError(f"{name} component values must be in [0, 1]")
    
    def to_array(self) -> np.ndarray:
        """Convert to array format [T, I, F] for each sample."""
        return np.column_stack([self.truth, self.indeterminacy, self.falsity])
    
    def get_feature_names(self) -> list:
        """Get feature names for the neutrosophic components."""
        return ['truth', 'indeterminacy', 'falsity']


class NeutrosophicTransformer:
    """
    Neutrosophic transformation implementation based on Definition 3 from the paper.
    
    Transforms dual clustering outputs into neutrosophic components (T, I, F) where:
    - Truth (T): Degree of certainty in primary cluster assignment
    - Indeterminacy (I): Structural ambiguity based on FCM membership entropy
    - Falsity (F): Degree of evidence against primary assignment
    """
    
    def __init__(self, entropy_epsilon: float = 1e-9, entropy_base: float = 2.0):
        """Initialize neutrosophic transformer.
        
        Args:
            entropy_epsilon: Small constant for numerical stability in entropy calculation
            entropy_base: Base for entropy calculation (default: 2 for bits)
        """
        self.entropy_epsilon = entropy_epsilon
        self.entropy_base = entropy_base
        self.is_fitted = False
        
    def transform(self, kmeans_labels: np.ndarray, fcm_memberships: np.ndarray) -> NeutrosophicComponents:
        """Transform dual clustering outputs to neutrosophic components.
        
        Implementation of Definition 3 from the paper:
        - T(y_i) = u_{i,k_i} (FCM membership for K-means assigned cluster)
        - F(y_i) = 1 - T(y_i) (Complement of truth)
        - I(y_i) = H(u_i) / log_2(C) (Normalized Shannon entropy)
        
        Args:
            kmeans_labels: K-means cluster assignments of shape (n_samples,)
            fcm_memberships: FCM membership matrix of shape (n_samples, n_clusters)
            
        Returns:
            NeutrosophicComponents containing T, I, F arrays
        """
        self._validate_inputs(kmeans_labels, fcm_memberships)
        
        n_samples, n_clusters = fcm_memberships.shape
        
        logger.info(f"Transforming dual clustering outputs to neutrosophic components for {n_samples} samples")
        
        # Initialize component arrays
        truth = np.zeros(n_samples)
        indeterminacy = np.zeros(n_samples)
        falsity = np.zeros(n_samples)
        
        # Compute neutrosophic components for each sample
        for i in range(n_samples):
            # Get K-means assigned cluster for sample i
            kmeans_cluster = kmeans_labels[i]
            
            # Truth: FCM membership for K-means assigned cluster
            truth[i] = fcm_memberships[i, kmeans_cluster]
            
            # Falsity: Complement of truth (sum of memberships to other clusters)
            falsity[i] = 1.0 - truth[i]
            
            # Indeterminacy: Normalized Shannon entropy of FCM membership distribution
            membership_vector = fcm_memberships[i, :]
            entropy = compute_shannon_entropy(membership_vector, base=self.entropy_base, epsilon=self.entropy_epsilon)
            indeterminacy[i] = normalize_entropy(entropy, n_clusters, base=self.entropy_base)
        
        # Create neutrosophic components
        components = NeutrosophicComponents(
            truth=truth,
            indeterminacy=indeterminacy,
            falsity=falsity
        )
        
        logger.info("Neutrosophic transformation completed")
        logger.info(f"Truth range: [{np.min(truth):.3f}, {np.max(truth):.3f}]")
        logger.info(f"Indeterminacy range: [{np.min(indeterminacy):.3f}, {np.max(indeterminacy):.3f}]")
        logger.info(f"Falsity range: [{np.min(falsity):.3f}, {np.max(falsity):.3f}]")
        
        return components
    
    def fit_transform(self, kmeans_labels: np.ndarray, fcm_memberships: np.ndarray) -> NeutrosophicComponents:
        """Fit transformer and transform data (for consistency with sklearn API)."""
        self.is_fitted = True
        return self.transform(kmeans_labels, fcm_memberships)
    
    def create_enriched_features(self, original_features: np.ndarray, 
                               integrated_cluster_features: np.ndarray,
                               neutrosophic_components: NeutrosophicComponents) -> np.ndarray:
        """Create enriched feature set combining original, cluster, and neutrosophic features.
        
        Args:
            original_features: Original input features
            integrated_cluster_features: Dual clustering features [one_hot_kmeans, fcm_memberships]
            neutrosophic_components: Neutrosophic components (T, I, F)
            
        Returns:
            Enriched feature matrix
        """
        # Convert neutrosophic components to array
        neutrosophic_array = neutrosophic_components.to_array()
        
        # Concatenate all features
        enriched_features = np.concatenate([
            original_features,
            integrated_cluster_features,
            neutrosophic_array
        ], axis=1)
        
        logger.info(f"Created enriched features with shape {enriched_features.shape}")
        
        return enriched_features
    
    def get_feature_names(self, original_feature_names: list, n_clusters: int) -> list:
        """Get feature names for the enriched feature set.
        
        Args:
            original_feature_names: Names of original features
            n_clusters: Number of clusters
            
        Returns:
            List of all feature names
        """
        # Original feature names
        feature_names = original_feature_names.copy()
        
        # K-means one-hot feature names
        feature_names.extend([f'kmeans_cluster_{i}' for i in range(n_clusters)])
        
        # FCM membership feature names
        feature_names.extend([f'fcm_membership_{i}' for i in range(n_clusters)])
        
        # Neutrosophic component names
        feature_names.extend(['truth', 'indeterminacy', 'falsity'])
        
        return feature_names
    
    def analyze_neutrosophic_distribution(self, components: NeutrosophicComponents) -> Dict[str, Any]:
        """Analyze the distribution of neutrosophic components.
        
        Args:
            components: Neutrosophic components to analyze
            
        Returns:
            Dictionary with distribution statistics
        """
        analysis = {}
        
        for component_name, component_values in [
            ('truth', components.truth),
            ('indeterminacy', components.indeterminacy),
            ('falsity', components.falsity)
        ]:
            analysis[component_name] = {
                'mean': np.mean(component_values),
                'std': np.std(component_values),
                'min': np.min(component_values),
                'max': np.max(component_values),
                'median': np.median(component_values),
                'q25': np.percentile(component_values, 25),
                'q75': np.percentile(component_values, 75)
            }
        
        # Additional analysis
        analysis['correlations'] = {
            'truth_indeterminacy': np.corrcoef(components.truth, components.indeterminacy)[0, 1],
            'truth_falsity': np.corrcoef(components.truth, components.falsity)[0, 1],
            'indeterminacy_falsity': np.corrcoef(components.indeterminacy, components.falsity)[0, 1]
        }
        
        # High indeterminacy points (potential transition regions)
        high_indeterminacy_threshold = np.percentile(components.indeterminacy, 90)
        high_indeterminacy_ratio = np.mean(components.indeterminacy > high_indeterminacy_threshold)
        
        analysis['high_indeterminacy'] = {
            'threshold': high_indeterminacy_threshold,
            'ratio': high_indeterminacy_ratio,
            'count': np.sum(components.indeterminacy > high_indeterminacy_threshold)
        }
        
        return analysis
    
    def _validate_inputs(self, kmeans_labels: np.ndarray, fcm_memberships: np.ndarray) -> None:
        """Validate inputs for neutrosophic transformation."""
        # Check K-means labels
        if not isinstance(kmeans_labels, np.ndarray):
            raise TypeError("kmeans_labels must be a numpy array")
        
        if kmeans_labels.ndim != 1:
            raise ValueError("kmeans_labels must be 1-dimensional")
        
        if len(kmeans_labels) == 0:
            raise ValueError("kmeans_labels cannot be empty")
        
        # Check FCM memberships
        if not isinstance(fcm_memberships, np.ndarray):
            raise TypeError("fcm_memberships must be a numpy array")
        
        if fcm_memberships.ndim != 2:
            raise ValueError("fcm_memberships must be 2-dimensional")
        
        if fcm_memberships.shape[0] != len(kmeans_labels):
            raise ValueError("Number of samples in kmeans_labels and fcm_memberships must match")
        
        # Check membership matrix properties
        if np.any(fcm_memberships < 0) or np.any(fcm_memberships > 1):
            raise ValueError("FCM memberships must be in [0, 1]")
        
        # Check if rows sum to approximately 1 (with tolerance for numerical errors)
        row_sums = np.sum(fcm_memberships, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            logger.warning("FCM membership rows do not sum to 1.0 (may cause issues)")
        
        # Check K-means label range
        n_clusters = fcm_memberships.shape[1]
        if np.any(kmeans_labels < 0) or np.any(kmeans_labels >= n_clusters):
            raise ValueError(f"K-means labels must be in range [0, {n_clusters-1}]")
    
    def get_params(self) -> Dict[str, Any]:
        """Get transformer parameters."""
        return {
            'entropy_epsilon': self.entropy_epsilon,
            'entropy_base': self.entropy_base
        }
    
    def set_params(self, **params) -> 'NeutrosophicTransformer':
        """Set transformer parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self