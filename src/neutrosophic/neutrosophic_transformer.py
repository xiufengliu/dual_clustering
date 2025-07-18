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
        # Ensure parameters are proper numeric types (handle YAML string loading)
        self.entropy_epsilon = float(entropy_epsilon) if entropy_epsilon is not None else 1e-9
        self.entropy_base = float(entropy_base) if entropy_base is not None else 2.0
        self.is_fitted = False

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Log the actual values for debugging
        self.logger.debug(f"NeutrosophicTransformer initialized with entropy_epsilon={self.entropy_epsilon} (type: {type(self.entropy_epsilon)}), entropy_base={self.entropy_base} (type: {type(self.entropy_base)})")
        
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
        logger.info("Starting neutrosophic transformation with robust data type handling")
        
        # Step 1: Robust input conversion and validation
        try:
            # Force conversion to numpy arrays with specific dtypes
            kmeans_labels = np.asarray(kmeans_labels, dtype=np.int32)
            fcm_memberships = np.asarray(fcm_memberships, dtype=np.float64)
            
            logger.debug(f"Converted input types - kmeans_labels: {kmeans_labels.dtype}, fcm_memberships: {fcm_memberships.dtype}")
            logger.debug(f"Converted input shapes - kmeans_labels: {kmeans_labels.shape}, fcm_memberships: {fcm_memberships.shape}")
            
            # Validate basic properties
            if kmeans_labels.ndim != 1:
                raise ValueError("kmeans_labels must be 1-dimensional")
            if fcm_memberships.ndim != 2:
                raise ValueError("fcm_memberships must be 2-dimensional")
            if len(kmeans_labels) != fcm_memberships.shape[0]:
                raise ValueError("Number of samples in kmeans_labels and fcm_memberships must match")
                
            # Validate K-means labels range
            n_clusters_fcm = fcm_memberships.shape[1]
            if np.any(kmeans_labels < 0) or np.any(kmeans_labels >= n_clusters_fcm):
                raise ValueError(f"K-means labels must be in range [0, {n_clusters_fcm-1}], but found min={np.min(kmeans_labels)} and max={np.max(kmeans_labels)}")

            # Validate FCM memberships sum to approx 1
            row_sums = np.sum(fcm_memberships, axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                logger.warning("FCM membership rows do not sum to 1.0 (may indicate issues or non-normalized data)")
                # Optionally, re-normalize if not summing to 1.0
                fcm_memberships = fcm_memberships / row_sums[:, np.newaxis]
                fcm_memberships = np.nan_to_num(fcm_memberships, nan=1.0/n_clusters_fcm) # Handle division by zero if a row sum was 0
                logger.warning("FCM memberships re-normalized to sum to 1.0")

        except Exception as e:
            logger.error(f"Input validation and conversion failed: {e}")
            raise ValueError(f"Failed to prepare inputs for neutrosophic transformation: {e}") from e
        
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
        logger.info("Creating enriched features from original, cluster, and neutrosophic components")

        try:
            # Step 1: Comprehensive data type validation and conversion
            logger.debug(f"Input data types - original: {original_features.dtype}, integrated: {integrated_cluster_features.dtype}")

            # Convert all inputs to float64 with robust error handling
            original_features = self._safe_convert_to_float64(original_features, "original_features")
            integrated_cluster_features = self._safe_convert_to_float64(integrated_cluster_features, "integrated_cluster_features")
            neutrosophic_array = self._safe_convert_to_float64(neutrosophic_components.to_array(), "neutrosophic_components")

            # Additional validation: ensure all arrays are 2D
            if original_features.ndim == 1:
                original_features = original_features.reshape(-1, 1)
            if integrated_cluster_features.ndim == 1:
                integrated_cluster_features = integrated_cluster_features.reshape(-1, 1)
            if neutrosophic_array.ndim == 1:
                neutrosophic_array = neutrosophic_array.reshape(-1, 1)

            # Step 2: Validate shapes
            n_samples = original_features.shape[0]
            if integrated_cluster_features.shape[0] != n_samples:
                raise ValueError(f"Sample count mismatch: original={n_samples}, integrated={integrated_cluster_features.shape[0]}")
            if neutrosophic_array.shape[0] != n_samples:
                raise ValueError(f"Sample count mismatch: original={n_samples}, neutrosophic={neutrosophic_array.shape[0]}")

            logger.debug(f"Feature shapes - original: {original_features.shape}, integrated: {integrated_cluster_features.shape}, neutrosophic: {neutrosophic_array.shape}")

            # Step 3: Safe concatenation with explicit dtype control
            feature_list = []

            # Add original features
            if original_features.size > 0:
                feature_list.append(original_features.astype(np.float64))

            # Add integrated cluster features
            if integrated_cluster_features.size > 0:
                feature_list.append(integrated_cluster_features.astype(np.float64))

            # Add neutrosophic features
            if neutrosophic_array.size > 0:
                feature_list.append(neutrosophic_array.astype(np.float64))

            # Concatenate with explicit dtype specification
            if len(feature_list) > 1:
                enriched_features = np.concatenate(feature_list, axis=1, dtype=np.float64)
            elif len(feature_list) == 1:
                enriched_features = feature_list[0].astype(np.float64)
            else:
                # Fallback: create minimal feature matrix
                enriched_features = np.zeros((n_samples, 1), dtype=np.float64)
                logger.warning("No valid features found, created zero matrix")

            # Step 4: Final validation and cleanup
            if not np.all(np.isfinite(enriched_features)):
                logger.warning("Non-finite values detected in enriched features, replacing with zeros")
                enriched_features = np.where(np.isfinite(enriched_features), enriched_features, 0.0)

            # Ensure final result is float64
            enriched_features = enriched_features.astype(np.float64)

            logger.info(f"Successfully created enriched features: shape={enriched_features.shape}, dtype={enriched_features.dtype}")
            return enriched_features

        except Exception as e:
            logger.error(f"Feature enrichment failed: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")

            # Enhanced fallback with better error handling
            try:
                fallback_features = self._safe_convert_to_float64(original_features, "fallback_original_features")
                if fallback_features.ndim == 1:
                    fallback_features = fallback_features.reshape(-1, 1)
                logger.warning(f"Using fallback features with shape: {fallback_features.shape}")
                return fallback_features
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                # Ultimate fallback: create zero matrix
                n_samples = len(original_features) if hasattr(original_features, '__len__') else 1
                zero_features = np.zeros((n_samples, 1), dtype=np.float64)
                logger.warning(f"Using zero matrix fallback with shape: {zero_features.shape}")
                return zero_features
    
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

    def _safe_convert_to_float64(self, data: np.ndarray, data_name: str) -> np.ndarray:
        """Safely convert data to float64, handling mixed data types.

        Args:
            data: Input data array
            data_name: Name of the data for logging

        Returns:
            Float64 numpy array
        """
        try:
            # Handle None or empty data
            if data is None:
                logger.warning(f"{data_name} is None, creating zero array")
                return np.zeros((1, 1), dtype=np.float64)

            # First, ensure it's a numpy array
            data = np.asarray(data)

            # Handle empty arrays
            if data.size == 0:
                logger.warning(f"{data_name} is empty, creating minimal zero array")
                return np.zeros((1, 1), dtype=np.float64)

            logger.debug(f"Converting {data_name}: shape={data.shape}, dtype={data.dtype}, kind={data.dtype.kind}")

            # Check if data contains string/object types
            if data.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
                logger.warning(f"{data_name} contains non-numeric data types: {data.dtype}")

                # Sample some values for debugging
                flat_data = data.flatten()
                sample_values = flat_data[:min(5, len(flat_data))]
                logger.debug(f"Sample values in {data_name}: {sample_values}")

                # Try to convert each element to float, replacing invalid values with 0.0
                converted_data = np.zeros(data.shape, dtype=np.float64)
                flat_converted = converted_data.flatten()

                invalid_count = 0
                for i, value in enumerate(flat_data):
                    try:
                        if isinstance(value, (str, bytes)):
                            # Handle common string representations
                            if isinstance(value, str):
                                value = value.strip()
                                if value.lower() in ['nan', 'none', 'null', '']:
                                    flat_converted[i] = 0.0
                                    invalid_count += 1
                                else:
                                    flat_converted[i] = float(value)
                            else:
                                # Handle bytes
                                flat_converted[i] = float(value.decode('utf-8'))
                        elif np.isscalar(value):
                            flat_converted[i] = float(value)
                        else:
                            # Handle complex objects
                            flat_converted[i] = float(str(value))
                    except (ValueError, TypeError, UnicodeDecodeError):
                        # Replace invalid values with 0.0
                        flat_converted[i] = 0.0
                        invalid_count += 1
                        if invalid_count <= 10:  # Log first 10 invalid values
                            logger.debug(f"Replaced invalid value '{value}' (type: {type(value)}) with 0.0 in {data_name}")

                converted_data = flat_converted.reshape(data.shape)
                if invalid_count > 0:
                    logger.warning(f"Replaced {invalid_count} invalid values with 0.0 in {data_name}")
                logger.info(f"Converted {data_name} from {data.dtype} to float64")
                return converted_data

            elif data.dtype.kind in ['c']:  # Complex numbers
                logger.warning(f"{data_name} contains complex numbers, taking real part")
                return np.real(data).astype(np.float64)

            elif data.dtype.kind in ['b']:  # Boolean
                logger.info(f"{data_name} contains boolean data, converting to 0/1")
                return data.astype(np.float64)

            else:
                # Data is already numeric, just convert to float64
                numeric_data = data.astype(np.float64)

                # Check for any non-finite values
                if not np.all(np.isfinite(numeric_data)):
                    logger.warning(f"{data_name} contains non-finite values, replacing with 0.0")
                    numeric_data = np.where(np.isfinite(numeric_data), numeric_data, 0.0)

                return numeric_data

        except Exception as e:
            logger.error(f"Failed to convert {data_name} to float64: {e}")
            logger.error(f"Data info: shape={getattr(data, 'shape', 'unknown')}, dtype={getattr(data, 'dtype', 'unknown')}")

            # Enhanced fallback: try to determine appropriate shape
            try:
                if hasattr(data, 'shape') and data.shape:
                    fallback_shape = data.shape
                elif hasattr(data, '__len__'):
                    fallback_shape = (len(data),)
                else:
                    fallback_shape = (1,)

                logger.warning(f"Creating zero array as fallback for {data_name} with shape {fallback_shape}")
                return np.zeros(fallback_shape, dtype=np.float64)

            except Exception as fallback_error:
                logger.error(f"Fallback shape determination failed: {fallback_error}")
                logger.warning(f"Using minimal fallback array for {data_name}")
                return np.zeros((1, 1), dtype=np.float64)

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