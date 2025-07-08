"""Statistical significance testing for forecasting model comparison."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare
import warnings
import logging

logger = logging.getLogger(__name__)

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("Statsmodels not available for advanced statistical tests")


class StatisticalTests:
    """Comprehensive statistical testing framework for forecasting evaluation."""
    
    def __init__(self, alpha: float = 0.05):
        """Initialize statistical tests.
        
        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha
        
    def diebold_mariano_test(self, errors1: np.ndarray, errors2: np.ndarray, 
                           h: int = 1, power: int = 2) -> Dict[str, float]:
        """
        Diebold-Mariano test for comparing forecast accuracy.
        
        Tests the null hypothesis that two forecasts have equal accuracy.
        
        Args:
            errors1: Forecast errors from model 1
            errors2: Forecast errors from model 2  
            h: Forecast horizon
            power: Power for loss function (1=MAE, 2=MSE)
            
        Returns:
            Dictionary with test statistic, p-value, and conclusion
        """
        # Calculate loss differential
        loss1 = np.abs(errors1) ** power
        loss2 = np.abs(errors2) ** power
        d = loss1 - loss2
        
        # Mean of loss differential
        d_mean = np.mean(d)
        
        # Calculate variance of loss differential with autocorrelation correction
        n = len(d)
        d_var = np.var(d, ddof=1)
        
        # Autocorrelation correction for multi-step forecasts
        if h > 1:
            # Calculate autocorrelations up to h-1 lags
            autocorr_sum = 0
            for k in range(1, h):
                if n > k:
                    autocorr_k = np.corrcoef(d[:-k], d[k:])[0, 1]
                    if not np.isnan(autocorr_k):
                        autocorr_sum += autocorr_k
            
            d_var = d_var * (1 + 2 * autocorr_sum) / n
        else:
            d_var = d_var / n
            
        # Test statistic
        if d_var > 0:
            dm_stat = d_mean / np.sqrt(d_var)
        else:
            dm_stat = 0
            
        # P-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        
        # Conclusion
        significant = p_value < self.alpha
        
        return {
            'statistic': dm_stat,
            'p_value': p_value,
            'significant': significant,
            'alpha': self.alpha,
            'conclusion': 'Model 1 significantly different' if significant else 'No significant difference'
        }
    
    def modified_diebold_mariano_test(self, errors1: np.ndarray, errors2: np.ndarray,
                                    h: int = 1, power: int = 2) -> Dict[str, float]:
        """
        Modified Diebold-Mariano test with small sample correction.
        
        Args:
            errors1: Forecast errors from model 1
            errors2: Forecast errors from model 2
            h: Forecast horizon
            power: Power for loss function
            
        Returns:
            Dictionary with test results
        """
        # Standard DM test
        dm_result = self.diebold_mariano_test(errors1, errors2, h, power)
        
        # Small sample correction
        n = len(errors1)
        correction_factor = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
        
        corrected_stat = dm_result['statistic'] * correction_factor
        corrected_p_value = 2 * (1 - stats.t.cdf(np.abs(corrected_stat), df=n-1))
        
        return {
            'statistic': corrected_stat,
            'p_value': corrected_p_value,
            'significant': corrected_p_value < self.alpha,
            'alpha': self.alpha,
            'original_statistic': dm_result['statistic'],
            'correction_factor': correction_factor,
            'conclusion': 'Model 1 significantly different' if corrected_p_value < self.alpha else 'No significant difference'
        }
    
    def wilcoxon_signed_rank_test(self, errors1: np.ndarray, errors2: np.ndarray) -> Dict[str, float]:
        """
        Wilcoxon signed-rank test for comparing forecast accuracy.
        Non-parametric alternative to paired t-test.
        
        Args:
            errors1: Forecast errors from model 1
            errors2: Forecast errors from model 2
            
        Returns:
            Dictionary with test results
        """
        # Calculate absolute errors
        abs_errors1 = np.abs(errors1)
        abs_errors2 = np.abs(errors2)
        
        # Perform Wilcoxon signed-rank test
        try:
            statistic, p_value = wilcoxon(abs_errors1, abs_errors2, alternative='two-sided')
            significant = p_value < self.alpha
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'significant': significant,
                'alpha': self.alpha,
                'conclusion': 'Significantly different' if significant else 'No significant difference'
            }
        except Exception as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'alpha': self.alpha,
                'conclusion': 'Test failed',
                'error': str(e)
            }
    
    def friedman_test(self, errors_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Friedman test for comparing multiple forecasting models.
        
        Args:
            errors_dict: Dictionary mapping model names to error arrays
            
        Returns:
            Dictionary with test results
        """
        if len(errors_dict) < 3:
            raise ValueError("Friedman test requires at least 3 models")
            
        # Convert to absolute errors
        abs_errors = {name: np.abs(errors) for name, errors in errors_dict.items()}
        
        # Ensure all error arrays have the same length
        min_length = min(len(errors) for errors in abs_errors.values())
        abs_errors = {name: errors[:min_length] for name, errors in abs_errors.items()}
        
        # Perform Friedman test
        try:
            error_arrays = list(abs_errors.values())
            statistic, p_value = friedmanchisquare(*error_arrays)
            significant = p_value < self.alpha
            
            # Calculate ranks for post-hoc analysis
            model_names = list(abs_errors.keys())
            ranks = self._calculate_friedman_ranks(abs_errors)
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'significant': significant,
                'alpha': self.alpha,
                'model_ranks': dict(zip(model_names, ranks)),
                'conclusion': 'Models significantly different' if significant else 'No significant difference'
            }
        except Exception as e:
            logger.warning(f"Friedman test failed: {e}")
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'alpha': self.alpha,
                'conclusion': 'Test failed',
                'error': str(e)
            }
    
    def _calculate_friedman_ranks(self, errors_dict: Dict[str, np.ndarray]) -> List[float]:
        """Calculate average ranks for Friedman test."""
        n_models = len(errors_dict)
        n_samples = len(next(iter(errors_dict.values())))
        
        # Create matrix of errors
        error_matrix = np.array([errors for errors in errors_dict.values()])
        
        # Calculate ranks for each sample
        ranks = np.zeros_like(error_matrix, dtype=float)
        for i in range(n_samples):
            sample_errors = error_matrix[:, i]
            sample_ranks = stats.rankdata(sample_errors)
            ranks[:, i] = sample_ranks
            
        # Calculate average ranks
        avg_ranks = np.mean(ranks, axis=1)
        return avg_ranks.tolist()
    
    def multiple_comparison_correction(self, p_values: List[float], 
                                     method: str = 'bonferroni') -> Dict[str, Any]:
        """
        Apply multiple comparison correction to p-values.
        
        Args:
            p_values: List of p-values from multiple tests
            method: Correction method ('bonferroni', 'holm', 'fdr_bh')
            
        Returns:
            Dictionary with corrected p-values and significance
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        if method == 'bonferroni':
            corrected_p = p_values * n_tests
            corrected_p = np.minimum(corrected_p, 1.0)
        elif method == 'holm':
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros_like(p_values)
            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = p_values[idx] * (n_tests - i)
            corrected_p = np.minimum(corrected_p, 1.0)
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR control
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros_like(p_values)
            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = p_values[idx] * n_tests / (i + 1)
            corrected_p = np.minimum(corrected_p, 1.0)
        else:
            raise ValueError(f"Unknown correction method: {method}")
            
        significant = corrected_p < self.alpha
        
        return {
            'original_p_values': p_values.tolist(),
            'corrected_p_values': corrected_p.tolist(),
            'significant': significant.tolist(),
            'method': method,
            'alpha': self.alpha,
            'n_tests': n_tests
        }
    
    def comprehensive_model_comparison(self, errors_dict: Dict[str, np.ndarray],
                                     reference_model: str = None) -> Dict[str, Any]:
        """
        Comprehensive statistical comparison of multiple forecasting models.
        
        Args:
            errors_dict: Dictionary mapping model names to error arrays
            reference_model: Name of reference model for pairwise comparisons
            
        Returns:
            Dictionary with comprehensive test results
        """
        results = {
            'models': list(errors_dict.keys()),
            'n_models': len(errors_dict),
            'n_samples': len(next(iter(errors_dict.values())))
        }
        
        # Friedman test for overall comparison
        if len(errors_dict) >= 3:
            results['friedman_test'] = self.friedman_test(errors_dict)
        
        # Pairwise comparisons
        model_names = list(errors_dict.keys())
        if reference_model and reference_model in model_names:
            # Compare all models against reference
            pairwise_results = {}
            p_values = []
            
            for model_name in model_names:
                if model_name != reference_model:
                    dm_result = self.modified_diebold_mariano_test(
                        errors_dict[reference_model], 
                        errors_dict[model_name]
                    )
                    pairwise_results[f"{reference_model}_vs_{model_name}"] = dm_result
                    p_values.append(dm_result['p_value'])
            
            # Multiple comparison correction
            if p_values:
                correction_result = self.multiple_comparison_correction(p_values, method='holm')
                results['pairwise_comparisons'] = pairwise_results
                results['multiple_comparison_correction'] = correction_result
        
        return results
