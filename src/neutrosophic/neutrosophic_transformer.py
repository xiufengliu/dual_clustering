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
        """Validate neutrosophic components after initialization and ensure float64 dtype."""
        # Convert all components to float64
        self.truth = np.asarray(self.truth, dtype=np.float64)
        self.indeterminacy = np.asarray(self.indeterminacy, dtype=np.float64)
        self.falsity = np.asarray(self.falsity, dtype=np.float64)

        if not (self.truth.shape == self.indeterminacy.shape == self.falsity.shape):
            raise ValueError("All neutrosophic components must have the same shape")

        # Check value ranges
        for component, name in [(self.truth, "Truth"), (self.indeterminacy, "Indeterminacy"), (self.falsity, "Falsity")]:
            if np.any(component < 0) or np.any(component > 1):
                raise ValueError(f"{name} component values must be in [0, 1]")
    
    def to_array(self) -> np.ndarray:
        """Convert to array format [T, I, F] for each sample."""
        # Ensure all components are float64 before stacking
        truth_float = np.asarray(self.truth, dtype=np.float64)
        indeterminacy_float = np.asarray(self.indeterminacy, dtype=np.float64)
        falsity_float = np.asarray(self.falsity, dtype=np.float64)

        return np.column_stack([truth_float, indeterminacy_float, falsity_float])
    
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
        # Validate and ensure proper data types
        kmeans_labels = self._validate_inputs(kmeans_labels, fcm_memberships)

        # Additional dtype validation and conversion
        kmeans_labels = self._ensure_numeric_array(kmeans_labels, "kmeans_labels").astype(int)
        fcm_memberships = self._ensure_numeric_array(fcm_memberships, "fcm_memberships")
        
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

            # Ensure kmeans_cluster is an integer (not boolean or other type)
            if not isinstance(kmeans_cluster, (int, np.integer)):
                try:
                    kmeans_cluster = int(kmeans_cluster)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Cannot convert kmeans_cluster to int for sample {i}: {kmeans_cluster}, type: {type(kmeans_cluster)}") from e

            # Validate cluster index
            if kmeans_cluster < 0 or kmeans_cluster >= n_clusters:
                raise ValueError(f"Invalid cluster index {kmeans_cluster} for sample {i}. Must be in range [0, {n_clusters-1}]")

            # Validate FCM memberships shape
            if fcm_memberships.shape[1] != n_clusters:
                raise ValueError(f"FCM memberships shape mismatch: expected {n_clusters} clusters, got {fcm_memberships.shape[1]}")

            # Truth: FCM membership for K-means assigned cluster
            try:
                truth[i] = fcm_memberships[i, kmeans_cluster]
            except IndexError as e:
                raise IndexError(f"Index error accessing fcm_memberships[{i}, {kmeans_cluster}]. FCM shape: {fcm_memberships.shape}") from e
            
            # Falsity: Complement of truth (sum of memberships to other clusters)
            falsity[i] = 1.0 - truth[i]
            
            # Indeterminacy: Normalized Shannon entropy of FCM membership distribution
            membership_vector = fcm_memberships[i, :]
            entropy = compute_shannon_entropy(membership_vector, base=self.entropy_base, epsilon=self.entropy_epsilon)
            indeterminacy[i] = normalize_entropy(entropy, n_clusters, base=self.entropy_base)
        
        # Ensure all components are float64 before creating NeutrosophicComponents
        truth = np.asarray(truth, dtype=np.float64)
        indeterminacy = np.asarray(indeterminacy, dtype=np.float64)
        falsity = np.asarray(falsity, dtype=np.float64)

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
        # Ensure all inputs are numeric arrays with consistent dtype
        original_features = self._ensure_numeric_array(original_features, "original_features")
        integrated_cluster_features = self._ensure_numeric_array(integrated_cluster_features, "integrated_cluster_features")

        # Convert neutrosophic components to array
        neutrosophic_array = neutrosophic_components.to_array()
        neutrosophic_array = self._ensure_numeric_array(neutrosophic_array, "neutrosophic_array")

        # Ensure all arrays have the same number of samples
        n_samples = original_features.shape[0]
        if integrated_cluster_features.shape[0] != n_samples:
            raise ValueError(f"Mismatch in number of samples: original_features={n_samples}, integrated_cluster_features={integrated_cluster_features.shape[0]}")
        if neutrosophic_array.shape[0] != n_samples:
            raise ValueError(f"Mismatch in number of samples: original_features={n_samples}, neutrosophic_array={neutrosophic_array.shape[0]}")

        # Debug logging before concatenation
        logger.debug(f"Before concatenation - Original: shape={original_features.shape}, dtype={original_features.dtype}")
        logger.debug(f"Before concatenation - Integrated: shape={integrated_cluster_features.shape}, dtype={integrated_cluster_features.dtype}")
        logger.debug(f"Before concatenation - Neutrosophic: shape={neutrosophic_array.shape}, dtype={neutrosophic_array.dtype}")

        # Concatenate all features with explicit dtype conversion and error handling
        try:
            # Force conversion to float64 with robust handling
            original_float = self._force_float64_conversion(original_features, "original_features")
            integrated_float = self._force_float64_conversion(integrated_cluster_features, "integrated_cluster_features")
            neutrosophic_float = self._force_float64_conversion(neutrosophic_array, "neutrosophic_array")

            # Additional validation before concatenation
            if original_float.dtype != np.float64:
                logger.error(f"Original features still not float64: {original_float.dtype}")
                original_float = np.asarray(original_float, dtype=np.float64)

            if integrated_float.dtype != np.float64:
                logger.error(f"Integrated features still not float64: {integrated_float.dtype}")
                integrated_float = np.asarray(integrated_float, dtype=np.float64)

            if neutrosophic_float.dtype != np.float64:
                logger.error(f"Neutrosophic features still not float64: {neutrosophic_float.dtype}")
                neutrosophic_float = np.asarray(neutrosophic_float, dtype=np.float64)

            # Final concatenation with explicit dtype specification
            enriched_features = np.concatenate([
                original_float,
                integrated_float,
                neutrosophic_float
            ], axis=1).astype(np.float64)

            logger.debug(f"Concatenation successful - Result: shape={enriched_features.shape}, dtype={enriched_features.dtype}")

        except Exception as e:
            logger.error(f"Failed to concatenate features. Shapes: original={original_features.shape}, "
                        f"integrated={integrated_cluster_features.shape}, neutrosophic={neutrosophic_array.shape}")
            logger.error(f"Data types: original={original_features.dtype}, "
                        f"integrated={integrated_cluster_features.dtype}, neutrosophic={neutrosophic_array.dtype}")

            # More detailed error logging
            logger.error(f"Original features sample: {original_features.flat[:5] if original_features.size > 0 else 'empty'}")
            logger.error(f"Integrated features sample: {integrated_cluster_features.flat[:5] if integrated_cluster_features.size > 0 else 'empty'}")
            logger.error(f"Neutrosophic features sample: {neutrosophic_array.flat[:5] if neutrosophic_array.size > 0 else 'empty'}")

            # Fallback: create features with only numeric data
            logger.warning("Attempting fallback feature creation with only numeric data")
            try:
                # Use only original features if concatenation fails
                enriched_features = self._force_float64_conversion(original_features, "original_features_fallback")
                logger.warning("Using only original features due to concatenation failure")
            except Exception as e2:
                raise ValueError(f"Feature concatenation and fallback both failed: {e}, {e2}") from e

        logger.info(f"Created enriched features with shape {enriched_features.shape} and dtype {enriched_features.dtype}")

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
    
    def _validate_inputs(self, kmeans_labels: np.ndarray, fcm_memberships: np.ndarray) -> np.ndarray:
        """Validate inputs for neutrosophic transformation.

        Returns:
            Corrected kmeans_labels array
        """
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
        
        # Ensure kmeans_labels are integers
        if kmeans_labels.dtype != np.int32 and kmeans_labels.dtype != np.int64:
            logger.warning(f"Converting kmeans_labels from {kmeans_labels.dtype} to int")
            kmeans_labels = kmeans_labels.astype(int)

        # Check K-means label range
        n_clusters = fcm_memberships.shape[1]
        if np.any(kmeans_labels < 0) or np.any(kmeans_labels >= n_clusters):
            raise ValueError(f"K-means labels must be in range [0, {n_clusters-1}]")

        return kmeans_labels

    def _ensure_numeric_array(self, array: np.ndarray, array_name: str) -> np.ndarray:
        """Ensure array is numeric and handle dtype conversion issues.

        Args:
            array: Input array to validate and convert
            array_name: Name of the array for error reporting

        Returns:
            Numeric array with consistent dtype

        Raises:
            ValueError: If array cannot be converted to numeric
        """
        if not isinstance(array, np.ndarray):
            try:
                array = np.array(array)
            except Exception as e:
                raise ValueError(f"{array_name} cannot be converted to numpy array: {e}") from e

        # Check if array contains non-numeric data
        if array.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
            logger.warning(f"{array_name} contains non-numeric data (dtype: {array.dtype}). Attempting conversion.")

            # Try to convert to numeric
            try:
                # Flatten, convert, then reshape
                original_shape = array.shape
                flat_array = array.flatten()

                # Convert each element to float, handling strings and other types
                numeric_values = []
                for item in flat_array:
                    if isinstance(item, (str, bytes)):
                        # Try to parse as number
                        try:
                            numeric_values.append(float(item))
                        except (ValueError, TypeError):
                            # If it's a string that can't be converted, use 0.0 as default
                            logger.warning(f"Cannot convert '{item}' to numeric, using 0.0")
                            numeric_values.append(0.0)
                    elif np.isnan(item) or np.isinf(item):
                        # Handle NaN and inf values
                        numeric_values.append(0.0)
                    else:
                        numeric_values.append(float(item))

                array = np.array(numeric_values).reshape(original_shape)

            except Exception as e:
                raise ValueError(f"Failed to convert {array_name} to numeric: {e}") from e

        # Ensure array is float64
        if array.dtype != np.float64:
            try:
                array = array.astype(np.float64)
            except Exception as e:
                raise ValueError(f"Failed to convert {array_name} to float64: {e}") from e

        # Check for any remaining non-finite values
        if not np.all(np.isfinite(array)):
            logger.warning(f"{array_name} contains non-finite values. Replacing with 0.0")
            array = np.where(np.isfinite(array), array, 0.0)

        return array

    def _force_float64_conversion(self, array: np.ndarray, array_name: str) -> np.ndarray:
        """Force conversion to float64 with robust error handling.

        Args:
            array: Input array to convert
            array_name: Name for error reporting

        Returns:
            Array converted to float64
        """
        if array.dtype == np.float64:
            return array

        try:
            # First attempt: direct conversion
            return array.astype(np.float64)
        except (ValueError, TypeError) as e:
            logger.warning(f"Direct conversion failed for {array_name}: {e}")

            # Second attempt: element-by-element conversion
            try:
                original_shape = array.shape
                flat_array = array.flatten()
                converted_values = []

                for i, item in enumerate(flat_array):
                    try:
                        if isinstance(item, (str, bytes)):
                            # Try to parse string as number
                            if isinstance(item, bytes):
                                item = item.decode('utf-8')
                            # Remove any non-numeric characters and try conversion
                            cleaned_item = ''.join(c for c in str(item) if c.isdigit() or c in '.-+eE')
                            if cleaned_item:
                                converted_values.append(float(cleaned_item))
                            else:
                                converted_values.append(0.0)
                        elif np.isscalar(item):
                            if np.isfinite(float(item)):
                                converted_values.append(float(item))
                            else:
                                converted_values.append(0.0)
                        else:
                            converted_values.append(0.0)
                    except (ValueError, TypeError, OverflowError):
                        logger.warning(f"Could not convert element {i} ({item}) in {array_name}, using 0.0")
                        converted_values.append(0.0)

                result = np.array(converted_values, dtype=np.float64).reshape(original_shape)
                logger.info(f"Successfully converted {array_name} using element-by-element conversion")
                return result

            except Exception as e2:
                logger.error(f"Element-by-element conversion failed for {array_name}: {e2}")

                # Final fallback: create zero array with same shape
                logger.warning(f"Creating zero array for {array_name} as final fallback")
                return np.zeros(array.shape, dtype=np.float64)

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