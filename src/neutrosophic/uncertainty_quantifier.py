"""Uncertainty quantification utilities for neutrosophic components."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from .neutrosophic_transformer import NeutrosophicComponents

logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """Quantifies different types of uncertainty using neutrosophic components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize uncertainty quantifier.
        
        Args:
            config: Configuration dictionary with uncertainty parameters
        """
        self.config = config or {}
        self.high_uncertainty_threshold = self.config.get('high_uncertainty_threshold', 0.7)
        self.low_certainty_threshold = self.config.get('low_certainty_threshold', 0.3)
        
    def quantify_uncertainty(self, components: NeutrosophicComponents) -> Dict[str, Any]:
        """Comprehensive uncertainty quantification using neutrosophic components.
        
        Args:
            components: Neutrosophic components (T, I, F)
            
        Returns:
            Dictionary with various uncertainty measures
        """
        logger.info("Quantifying uncertainty from neutrosophic components")
        
        uncertainty_metrics = {}
        
        # Basic component statistics
        uncertainty_metrics['component_stats'] = self._compute_component_statistics(components)
        
        # Uncertainty categories
        uncertainty_metrics['uncertainty_categories'] = self._categorize_uncertainty(components)
        
        # Composite uncertainty measures
        uncertainty_metrics['composite_measures'] = self._compute_composite_measures(components)
        
        # Temporal uncertainty patterns (if applicable)
        uncertainty_metrics['temporal_patterns'] = self._analyze_temporal_patterns(components)
        
        # Uncertainty distribution analysis
        uncertainty_metrics['distribution_analysis'] = self._analyze_uncertainty_distribution(components)
        
        return uncertainty_metrics
    
    def _compute_component_statistics(self, components: NeutrosophicComponents) -> Dict[str, Dict[str, float]]:
        """Compute basic statistics for each neutrosophic component."""
        stats = {}
        
        for name, values in [('truth', components.truth), 
                           ('indeterminacy', components.indeterminacy), 
                           ('falsity', components.falsity)]:
            stats[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'skewness': float(self._compute_skewness(values)),
                'kurtosis': float(self._compute_kurtosis(values))
            }
        
        return stats
    
    def _categorize_uncertainty(self, components: NeutrosophicComponents) -> Dict[str, Any]:
        """Categorize data points based on uncertainty levels."""
        n_samples = len(components.truth)
        
        # High indeterminacy points (structural ambiguity)
        high_indeterminacy_mask = components.indeterminacy > self.high_uncertainty_threshold
        high_indeterminacy_count = np.sum(high_indeterminacy_mask)
        
        # Low truth points (low confidence in primary assignment)
        low_truth_mask = components.truth < self.low_certainty_threshold
        low_truth_count = np.sum(low_truth_mask)
        
        # High falsity points (strong evidence against primary assignment)
        high_falsity_mask = components.falsity > self.high_uncertainty_threshold
        high_falsity_count = np.sum(high_falsity_mask)
        
        # Combined high uncertainty (high I and low T)
        high_uncertainty_mask = high_indeterminacy_mask & low_truth_mask
        high_uncertainty_count = np.sum(high_uncertainty_mask)
        
        # Low uncertainty (high T and low I)
        low_uncertainty_mask = (components.truth > (1 - self.low_certainty_threshold)) & \
                              (components.indeterminacy < self.low_certainty_threshold)
        low_uncertainty_count = np.sum(low_uncertainty_mask)
        
        categories = {
            'high_indeterminacy': {
                'count': int(high_indeterminacy_count),
                'ratio': float(high_indeterminacy_count / n_samples),
                'indices': np.where(high_indeterminacy_mask)[0].tolist()
            },
            'low_truth': {
                'count': int(low_truth_count),
                'ratio': float(low_truth_count / n_samples),
                'indices': np.where(low_truth_mask)[0].tolist()
            },
            'high_falsity': {
                'count': int(high_falsity_count),
                'ratio': float(high_falsity_count / n_samples),
                'indices': np.where(high_falsity_mask)[0].tolist()
            },
            'high_uncertainty': {
                'count': int(high_uncertainty_count),
                'ratio': float(high_uncertainty_count / n_samples),
                'indices': np.where(high_uncertainty_mask)[0].tolist()
            },
            'low_uncertainty': {
                'count': int(low_uncertainty_count),
                'ratio': float(low_uncertainty_count / n_samples),
                'indices': np.where(low_uncertainty_mask)[0].tolist()
            }
        }
        
        return categories
    
    def _compute_composite_measures(self, components: NeutrosophicComponents) -> Dict[str, np.ndarray]:
        """Compute composite uncertainty measures."""
        # Overall uncertainty: combination of indeterminacy and low truth
        overall_uncertainty = components.indeterminacy * (1 - components.truth)
        
        # Conflict measure: when truth and falsity are both high (shouldn't happen in our formulation)
        conflict = np.minimum(components.truth, components.falsity)
        
        # Ignorance measure: when all components are low
        ignorance = 1 - (components.truth + components.indeterminacy + components.falsity)
        
        # Certainty measure: high truth with low indeterminacy
        certainty = components.truth * (1 - components.indeterminacy)
        
        # Ambiguity measure: primarily based on indeterminacy
        ambiguity = components.indeterminacy
        
        composite_measures = {
            'overall_uncertainty': overall_uncertainty,
            'conflict': conflict,
            'ignorance': ignorance,
            'certainty': certainty,
            'ambiguity': ambiguity
        }
        
        # Convert to regular arrays and compute statistics
        for measure_name, measure_values in composite_measures.items():
            composite_measures[f'{measure_name}_stats'] = {
                'mean': float(np.mean(measure_values)),
                'std': float(np.std(measure_values)),
                'min': float(np.min(measure_values)),
                'max': float(np.max(measure_values))
            }
        
        return composite_measures
    
    def _analyze_temporal_patterns(self, components: NeutrosophicComponents) -> Dict[str, Any]:
        """Analyze temporal patterns in uncertainty (if data has temporal structure)."""
        n_samples = len(components.truth)
        
        # Compute moving averages for uncertainty components
        window_sizes = [5, 10, 20] if n_samples > 20 else [min(5, n_samples//2)]
        
        temporal_patterns = {}
        
        for window_size in window_sizes:
            if window_size >= n_samples:
                continue
                
            # Moving average of indeterminacy
            indeterminacy_ma = self._moving_average(components.indeterminacy, window_size)
            
            # Moving average of truth
            truth_ma = self._moving_average(components.truth, window_size)
            
            # Volatility (moving standard deviation)
            indeterminacy_vol = self._moving_std(components.indeterminacy, window_size)
            
            temporal_patterns[f'window_{window_size}'] = {
                'indeterminacy_ma_mean': float(np.mean(indeterminacy_ma)),
                'indeterminacy_ma_std': float(np.std(indeterminacy_ma)),
                'truth_ma_mean': float(np.mean(truth_ma)),
                'truth_ma_std': float(np.std(truth_ma)),
                'indeterminacy_volatility_mean': float(np.mean(indeterminacy_vol)),
                'indeterminacy_volatility_std': float(np.std(indeterminacy_vol))
            }
        
        return temporal_patterns
    
    def _analyze_uncertainty_distribution(self, components: NeutrosophicComponents) -> Dict[str, Any]:
        """Analyze the distribution of uncertainty across the dataset."""
        # Histogram analysis for indeterminacy (primary uncertainty measure)
        hist_bins = 10
        indeterminacy_hist, bin_edges = np.histogram(components.indeterminacy, bins=hist_bins)
        
        # Find modes and distribution characteristics
        max_bin_idx = np.argmax(indeterminacy_hist)
        mode_range = (bin_edges[max_bin_idx], bin_edges[max_bin_idx + 1])
        
        # Concentration analysis
        high_uncertainty_ratio = np.mean(components.indeterminacy > 0.7)
        low_uncertainty_ratio = np.mean(components.indeterminacy < 0.3)
        
        distribution_analysis = {
            'indeterminacy_histogram': {
                'counts': indeterminacy_hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'mode_range': mode_range,
                'mode_count': int(indeterminacy_hist[max_bin_idx])
            },
            'concentration': {
                'high_uncertainty_ratio': float(high_uncertainty_ratio),
                'low_uncertainty_ratio': float(low_uncertainty_ratio),
                'medium_uncertainty_ratio': float(1 - high_uncertainty_ratio - low_uncertainty_ratio)
            },
            'distribution_shape': {
                'is_bimodal': self._check_bimodal(indeterminacy_hist),
                'is_uniform': self._check_uniform(indeterminacy_hist),
                'is_concentrated': high_uncertainty_ratio > 0.8 or low_uncertainty_ratio > 0.8
            }
        }
        
        return distribution_analysis
    
    def get_uncertainty_weights(self, components: NeutrosophicComponents, 
                              method: str = 'indeterminacy') -> np.ndarray:
        """Get uncertainty weights for prediction interval construction.
        
        Args:
            components: Neutrosophic components
            method: Method for computing weights ('indeterminacy', 'composite', 'truth_based')
            
        Returns:
            Uncertainty weights array
        """
        if method == 'indeterminacy':
            # Use indeterminacy directly as uncertainty weight
            weights = components.indeterminacy
            
        elif method == 'composite':
            # Composite uncertainty: indeterminacy weighted by low truth
            weights = components.indeterminacy * (1 - components.truth)
            
        elif method == 'truth_based':
            # Inverse of truth (low truth = high uncertainty)
            weights = 1 - components.truth
            
        else:
            raise ValueError(f"Unknown uncertainty weight method: {method}")
        
        # Normalize weights to [0, 1]
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        
        return weights
    
    def identify_transition_regions(self, components: NeutrosophicComponents, 
                                  threshold: float = 0.7) -> Dict[str, Any]:
        """Identify potential transition regions based on high indeterminacy.
        
        Args:
            components: Neutrosophic components
            threshold: Threshold for high indeterminacy
            
        Returns:
            Information about transition regions
        """
        high_indeterminacy_mask = components.indeterminacy > threshold
        transition_indices = np.where(high_indeterminacy_mask)[0]
        
        # Find consecutive regions
        if len(transition_indices) > 0:
            # Find breaks in consecutive indices
            breaks = np.where(np.diff(transition_indices) > 1)[0]
            
            # Create regions
            regions = []
            start_idx = 0
            
            for break_idx in breaks:
                end_idx = break_idx + 1
                region_start = transition_indices[start_idx]
                region_end = transition_indices[end_idx - 1]
                regions.append((region_start, region_end))
                start_idx = end_idx
            
            # Add final region
            if start_idx < len(transition_indices):
                region_start = transition_indices[start_idx]
                region_end = transition_indices[-1]
                regions.append((region_start, region_end))
        else:
            regions = []
        
        transition_info = {
            'total_transition_points': len(transition_indices),
            'transition_ratio': len(transition_indices) / len(components.indeterminacy),
            'transition_indices': transition_indices.tolist(),
            'transition_regions': regions,
            'num_regions': len(regions),
            'avg_region_length': np.mean([end - start + 1 for start, end in regions]) if regions else 0
        }
        
        return transition_info
    
    def _moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Compute moving average."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def _moving_std(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Compute moving standard deviation."""
        moving_std = []
        for i in range(window_size - 1, len(data)):
            window_data = data[i - window_size + 1:i + 1]
            moving_std.append(np.std(window_data))
        return np.array(moving_std)
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return np.mean(((data - mean_val) / std_val) ** 4) - 3
    
    def _check_bimodal(self, histogram: np.ndarray) -> bool:
        """Check if histogram suggests bimodal distribution."""
        # Simple heuristic: look for two peaks with a valley between
        if len(histogram) < 3:
            return False
        
        peaks = []
        for i in range(1, len(histogram) - 1):
            if histogram[i] > histogram[i-1] and histogram[i] > histogram[i+1]:
                peaks.append(i)
        
        return len(peaks) >= 2
    
    def _check_uniform(self, histogram: np.ndarray, tolerance: float = 0.2) -> bool:
        """Check if histogram suggests uniform distribution."""
        expected_count = np.mean(histogram)
        relative_deviations = np.abs(histogram - expected_count) / expected_count
        return np.all(relative_deviations < tolerance)