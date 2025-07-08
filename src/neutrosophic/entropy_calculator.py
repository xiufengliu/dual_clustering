"""Entropy calculation utilities for neutrosophic transformations."""

import numpy as np
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class EntropyCalculator:
    """Calculator for various entropy measures used in neutrosophic transformations."""
    
    def __init__(self, epsilon: float = 1e-9, default_base: float = 2.0):
        """Initialize entropy calculator.
        
        Args:
            epsilon: Small constant for numerical stability
            default_base: Default base for logarithm calculations
        """
        self.epsilon = epsilon
        self.default_base = default_base
    
    def shannon_entropy(self, probabilities: np.ndarray, 
                       base: Optional[float] = None,
                       normalize: bool = False) -> Union[float, np.ndarray]:
        """Calculate Shannon entropy of probability distribution(s).
        
        Args:
            probabilities: Probability array(s). Can be 1D or 2D.
            base: Base for logarithm (default: self.default_base)
            normalize: Whether to normalize by maximum possible entropy
            
        Returns:
            Shannon entropy value(s)
        """
        if base is None:
            base = self.default_base
        
        # Handle both 1D and 2D arrays
        if probabilities.ndim == 1:
            return self._shannon_entropy_1d(probabilities, base, normalize)
        elif probabilities.ndim == 2:
            return self._shannon_entropy_2d(probabilities, base, normalize)
        else:
            raise ValueError("Probabilities array must be 1D or 2D")
    
    def _shannon_entropy_1d(self, probabilities: np.ndarray, base: float, 
                           normalize: bool) -> float:
        """Calculate Shannon entropy for 1D probability array."""
        # Add epsilon for numerical stability
        probs = probabilities + self.epsilon
        
        # Normalize to ensure sum = 1
        probs = probs / np.sum(probs)
        
        # Calculate entropy
        log_probs = np.log(probs) / np.log(base)
        entropy = -np.sum(probs * log_probs)
        
        # Normalize if requested
        if normalize:
            max_entropy = np.log(len(probabilities)) / np.log(base)
            if max_entropy > 0:
                entropy = entropy / max_entropy
        
        return entropy
    
    def _shannon_entropy_2d(self, probabilities: np.ndarray, base: float, 
                           normalize: bool) -> np.ndarray:
        """Calculate Shannon entropy for 2D probability array (row-wise)."""
        n_samples, n_classes = probabilities.shape
        entropies = np.zeros(n_samples)
        
        for i in range(n_samples):
            entropies[i] = self._shannon_entropy_1d(probabilities[i], base, normalize)
        
        return entropies
    
    def renyi_entropy(self, probabilities: np.ndarray, alpha: float = 2.0,
                     base: Optional[float] = None, normalize: bool = False) -> Union[float, np.ndarray]:
        """Calculate Rényi entropy of order alpha.
        
        Args:
            probabilities: Probability array(s)
            alpha: Order of Rényi entropy (alpha != 1)
            base: Base for logarithm
            normalize: Whether to normalize by maximum possible entropy
            
        Returns:
            Rényi entropy value(s)
        """
        if base is None:
            base = self.default_base
        
        if alpha == 1.0:
            logger.warning("Rényi entropy with alpha=1 is Shannon entropy")
            return self.shannon_entropy(probabilities, base, normalize)
        
        if probabilities.ndim == 1:
            return self._renyi_entropy_1d(probabilities, alpha, base, normalize)
        elif probabilities.ndim == 2:
            return self._renyi_entropy_2d(probabilities, alpha, base, normalize)
        else:
            raise ValueError("Probabilities array must be 1D or 2D")
    
    def _renyi_entropy_1d(self, probabilities: np.ndarray, alpha: float, 
                         base: float, normalize: bool) -> float:
        """Calculate Rényi entropy for 1D probability array."""
        # Add epsilon for numerical stability
        probs = probabilities + self.epsilon
        
        # Normalize to ensure sum = 1
        probs = probs / np.sum(probs)
        
        # Calculate Rényi entropy
        if alpha == np.inf:
            # Max entropy (min entropy)
            entropy = -np.log(np.max(probs)) / np.log(base)
        else:
            sum_powered = np.sum(np.power(probs, alpha))
            entropy = (1.0 / (1.0 - alpha)) * np.log(sum_powered) / np.log(base)
        
        # Normalize if requested
        if normalize:
            if alpha == np.inf:
                max_entropy = np.log(len(probabilities)) / np.log(base)
            else:
                # Maximum Rényi entropy for uniform distribution
                uniform_prob = 1.0 / len(probabilities)
                max_entropy = (1.0 / (1.0 - alpha)) * np.log(len(probabilities) * (uniform_prob ** alpha)) / np.log(base)
            
            if max_entropy > 0:
                entropy = entropy / max_entropy
        
        return entropy
    
    def _renyi_entropy_2d(self, probabilities: np.ndarray, alpha: float, 
                         base: float, normalize: bool) -> np.ndarray:
        """Calculate Rényi entropy for 2D probability array (row-wise)."""
        n_samples = probabilities.shape[0]
        entropies = np.zeros(n_samples)
        
        for i in range(n_samples):
            entropies[i] = self._renyi_entropy_1d(probabilities[i], alpha, base, normalize)
        
        return entropies
    
    def tsallis_entropy(self, probabilities: np.ndarray, q: float = 2.0,
                       normalize: bool = False) -> Union[float, np.ndarray]:
        """Calculate Tsallis entropy of order q.
        
        Args:
            probabilities: Probability array(s)
            q: Tsallis parameter (q != 1)
            normalize: Whether to normalize by maximum possible entropy
            
        Returns:
            Tsallis entropy value(s)
        """
        if q == 1.0:
            logger.warning("Tsallis entropy with q=1 approaches Shannon entropy")
            return self.shannon_entropy(probabilities, normalize=normalize)
        
        if probabilities.ndim == 1:
            return self._tsallis_entropy_1d(probabilities, q, normalize)
        elif probabilities.ndim == 2:
            return self._tsallis_entropy_2d(probabilities, q, normalize)
        else:
            raise ValueError("Probabilities array must be 1D or 2D")
    
    def _tsallis_entropy_1d(self, probabilities: np.ndarray, q: float, 
                           normalize: bool) -> float:
        """Calculate Tsallis entropy for 1D probability array."""
        # Add epsilon for numerical stability
        probs = probabilities + self.epsilon
        
        # Normalize to ensure sum = 1
        probs = probs / np.sum(probs)
        
        # Calculate Tsallis entropy
        sum_powered = np.sum(np.power(probs, q))
        entropy = (1.0 / (q - 1.0)) * (1.0 - sum_powered)
        
        # Normalize if requested
        if normalize:
            # Maximum Tsallis entropy for uniform distribution
            uniform_prob = 1.0 / len(probabilities)
            max_entropy = (1.0 / (q - 1.0)) * (1.0 - len(probabilities) * (uniform_prob ** q))
            
            if max_entropy > 0:
                entropy = entropy / max_entropy
        
        return entropy
    
    def _tsallis_entropy_2d(self, probabilities: np.ndarray, q: float, 
                           normalize: bool) -> np.ndarray:
        """Calculate Tsallis entropy for 2D probability array (row-wise)."""
        n_samples = probabilities.shape[0]
        entropies = np.zeros(n_samples)
        
        for i in range(n_samples):
            entropies[i] = self._tsallis_entropy_1d(probabilities[i], q, normalize)
        
        return entropies
    
    def gini_simpson_index(self, probabilities: np.ndarray) -> Union[float, np.ndarray]:
        """Calculate Gini-Simpson diversity index.
        
        Args:
            probabilities: Probability array(s)
            
        Returns:
            Gini-Simpson index value(s)
        """
        if probabilities.ndim == 1:
            return self._gini_simpson_1d(probabilities)
        elif probabilities.ndim == 2:
            return self._gini_simpson_2d(probabilities)
        else:
            raise ValueError("Probabilities array must be 1D or 2D")
    
    def _gini_simpson_1d(self, probabilities: np.ndarray) -> float:
        """Calculate Gini-Simpson index for 1D probability array."""
        # Normalize probabilities
        probs = probabilities / np.sum(probabilities)
        
        # Gini-Simpson index = 1 - sum(p_i^2)
        return 1.0 - np.sum(probs ** 2)
    
    def _gini_simpson_2d(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate Gini-Simpson index for 2D probability array (row-wise)."""
        n_samples = probabilities.shape[0]
        indices = np.zeros(n_samples)
        
        for i in range(n_samples):
            indices[i] = self._gini_simpson_1d(probabilities[i])
        
        return indices
    
    def normalized_entropy(self, probabilities: np.ndarray, 
                          entropy_type: str = 'shannon',
                          **kwargs) -> Union[float, np.ndarray]:
        """Calculate normalized entropy (entropy / max_possible_entropy).
        
        Args:
            probabilities: Probability array(s)
            entropy_type: Type of entropy ('shannon', 'renyi', 'tsallis')
            **kwargs: Additional arguments for specific entropy types
            
        Returns:
            Normalized entropy value(s)
        """
        if entropy_type == 'shannon':
            return self.shannon_entropy(probabilities, normalize=True, **kwargs)
        elif entropy_type == 'renyi':
            return self.renyi_entropy(probabilities, normalize=True, **kwargs)
        elif entropy_type == 'tsallis':
            return self.tsallis_entropy(probabilities, normalize=True, **kwargs)
        else:
            raise ValueError(f"Unknown entropy type: {entropy_type}")
    
    def entropy_based_uncertainty(self, membership_matrix: np.ndarray,
                                 method: str = 'shannon_normalized') -> np.ndarray:
        """Calculate uncertainty measures based on entropy of membership distributions.
        
        This is the main method used for neutrosophic indeterminacy calculation.
        
        Args:
            membership_matrix: FCM membership matrix of shape (n_samples, n_clusters)
            method: Entropy method to use
            
        Returns:
            Uncertainty values for each sample
        """
        if method == 'shannon_normalized':
            return self.shannon_entropy(membership_matrix, normalize=True)
        elif method == 'shannon':
            return self.shannon_entropy(membership_matrix, normalize=False)
        elif method == 'gini_simpson':
            return self.gini_simpson_index(membership_matrix)
        elif method == 'renyi_2':
            return self.renyi_entropy(membership_matrix, alpha=2.0, normalize=True)
        elif method == 'tsallis_2':
            return self.tsallis_entropy(membership_matrix, q=2.0, normalize=True)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
    
    def compare_entropy_methods(self, probabilities: np.ndarray) -> dict:
        """Compare different entropy methods on the same probability distribution(s).
        
        Args:
            probabilities: Probability array(s)
            
        Returns:
            Dictionary with entropy values from different methods
        """
        results = {}
        
        # Shannon entropy
        results['shannon'] = self.shannon_entropy(probabilities, normalize=False)
        results['shannon_normalized'] = self.shannon_entropy(probabilities, normalize=True)
        
        # Rényi entropy (different orders)
        for alpha in [0.5, 2.0, np.inf]:
            results[f'renyi_alpha_{alpha}'] = self.renyi_entropy(probabilities, alpha=alpha, normalize=True)
        
        # Tsallis entropy (different orders)
        for q in [0.5, 2.0]:
            results[f'tsallis_q_{q}'] = self.tsallis_entropy(probabilities, q=q, normalize=True)
        
        # Gini-Simpson index
        results['gini_simpson'] = self.gini_simpson_index(probabilities)
        
        return results