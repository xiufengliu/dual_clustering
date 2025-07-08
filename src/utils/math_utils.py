"""Mathematical utilities for the neutrosophic forecasting framework."""

import random
import numpy as np
from typing import Tuple, Optional
from sklearn.utils import check_random_state
import logging

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    # Note: sklearn uses numpy's random state
    logger.info(f"Set random seeds to {seed}")


def normalize_data(data: np.ndarray, method: str = "min_max") -> Tuple[np.ndarray, dict]:
    """Normalize data using specified method.
    
    Args:
        data: Input data array
        method: Normalization method ('min_max' or 'z_score')
        
    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    if method == "min_max":
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max == data_min:
            logger.warning("Data has zero variance, normalization may not be meaningful")
            normalized = np.zeros_like(data)
        else:
            normalized = (data - data_min) / (data_max - data_min)
        
        params = {"method": "min_max", "min": data_min, "max": data_max}
        
    elif method == "z_score":
        data_mean = np.mean(data)
        data_std = np.std(data)
        
        if data_std == 0:
            logger.warning("Data has zero variance, normalization may not be meaningful")
            normalized = np.zeros_like(data)
        else:
            normalized = (data - data_mean) / data_std
        
        params = {"method": "z_score", "mean": data_mean, "std": data_std}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_data(normalized_data: np.ndarray, params: dict) -> np.ndarray:
    """Denormalize data using stored parameters.
    
    Args:
        normalized_data: Normalized data array
        params: Normalization parameters from normalize_data
        
    Returns:
        Denormalized data array
    """
    method = params["method"]
    
    if method == "min_max":
        data_min = params["min"]
        data_max = params["max"]
        return normalized_data * (data_max - data_min) + data_min
        
    elif method == "z_score":
        data_mean = params["mean"]
        data_std = params["std"]
        return normalized_data * data_std + data_mean
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_shannon_entropy(probabilities: np.ndarray, base: float = 2.0, 
                          epsilon: float = 1e-9) -> float:
    """Compute Shannon entropy of a probability distribution.
    
    Args:
        probabilities: Probability distribution array
        base: Logarithm base for entropy calculation
        epsilon: Small constant for numerical stability
        
    Returns:
        Shannon entropy value
    """
    # Add epsilon for numerical stability
    probs = probabilities + epsilon
    
    # Normalize to ensure sum = 1
    probs = probs / np.sum(probs)
    
    # Compute entropy
    entropy = -np.sum(probs * np.log(probs) / np.log(base))
    
    return entropy


def normalize_entropy(entropy: float, n_classes: int, base: float = 2.0) -> float:
    """Normalize entropy to [0, 1] range.
    
    Args:
        entropy: Raw entropy value
        n_classes: Number of classes/clusters
        base: Logarithm base used for entropy calculation
        
    Returns:
        Normalized entropy in [0, 1]
    """
    max_entropy = np.log(n_classes) / np.log(base)
    
    if max_entropy == 0:
        return 0.0
    
    return entropy / max_entropy


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                default_value: float = 0.0) -> np.ndarray:
    """Perform safe division with handling of zero denominators.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        default_value: Value to use when denominator is zero
        
    Returns:
        Result of safe division
    """
    result = np.full_like(numerator, default_value, dtype=float)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def validate_probability_distribution(probs: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Validate that an array represents a valid probability distribution.
    
    Args:
        probs: Probability array to validate
        tolerance: Tolerance for sum validation
        
    Returns:
        True if valid probability distribution
    """
    # Check non-negative values
    if np.any(probs < 0):
        return False
    
    # Check sum approximately equals 1
    if abs(np.sum(probs) - 1.0) > tolerance:
        return False
    
    return True