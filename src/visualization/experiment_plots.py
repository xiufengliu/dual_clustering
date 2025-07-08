"""Visualization tools for experimental results and analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    logger.warning("scikit-learn not available for t-SNE visualization")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available for interactive plots")


class ExperimentVisualizer:
    """Comprehensive visualization tools for experimental results."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        
        # Set style
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Create output directory
        self.output_dir = Path("results/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_model_comparison(self, results: Dict[str, Any], metric: str = 'rmse',
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot model comparison across datasets.
        
        Args:
            results: Experiment results dictionary
            metric: Metric to plot
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 6))
        if len(results) == 1:
            axes = [axes]
        
        for i, (dataset, dataset_results) in enumerate(results.items()):
            model_names = []
            metric_values = []
            
            for model_name, model_results in dataset_results['model_results'].items():
                if 'point_metrics' in model_results and not model_results.get('failed', False):
                    model_names.append(model_name)
                    metric_values.append(model_results['point_metrics'][metric])
            
            # Sort by metric value
            sorted_data = sorted(zip(model_names, metric_values), key=lambda x: x[1])
            model_names, metric_values = zip(*sorted_data)
            
            # Create bar plot
            bars = axes[i].bar(range(len(model_names)), metric_values)
            
            # Highlight our method
            for j, name in enumerate(model_names):
                if 'NDC-RF' in name or 'neutrosophic' in name.lower():
                    bars[j].set_color('red')
                    bars[j].set_alpha(0.8)
            
            axes[i].set_title(f'{dataset.replace("_", " ").title()}')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_xticks(range(len(model_names)))
            axes[i].set_xticklabels(model_names, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_statistical_significance_heatmap(self, statistical_results: Dict[str, Any],
                                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot statistical significance test results as heatmap.
        
        Args:
            statistical_results: Statistical test results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract p-values for heatmap
        datasets = list(statistical_results.keys())
        models = []
        p_values_matrix = []
        
        for dataset in datasets:
            if 'pairwise_dm_tests' in statistical_results[dataset]:
                dataset_p_values = []
                dataset_models = []
                
                for comparison, test_result in statistical_results[dataset]['pairwise_dm_tests'].items():
                    model_name = comparison.split('_vs_')[1]
                    dataset_models.append(model_name)
                    dataset_p_values.append(test_result['p_value'])
                
                if not models:
                    models = dataset_models
                p_values_matrix.append(dataset_p_values)
        
        if p_values_matrix:
            # Convert to numpy array
            p_values_array = np.array(p_values_matrix)
            
            # Create significance matrix (1 for significant, 0 for not)
            significance_matrix = (p_values_array < 0.05).astype(int)
            
            # Create heatmap
            sns.heatmap(significance_matrix, 
                       xticklabels=models,
                       yticklabels=datasets,
                       annot=p_values_array,
                       fmt='.3f',
                       cmap='RdYlBu_r',
                       cbar_kws={'label': 'Statistical Significance'},
                       ax=ax)
            
            ax.set_title('Statistical Significance Test Results\n(NDC-RF vs. Baselines)')
            ax.set_xlabel('Baseline Models')
            ax.set_ylabel('Datasets')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, float],
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot feature importance from Random Forest.
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        feature_names, importance_scores = zip(*sorted_features)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(feature_names)), importance_scores)
        
        # Highlight neutrosophic features
        neutrosophic_features = ['truth', 'indeterminacy', 'falsity', 'T', 'I', 'F']
        for i, name in enumerate(feature_names):
            if any(nf in name.lower() for nf in neutrosophic_features):
                bars[i].set_color('red')
                bars[i].set_alpha(0.8)
        
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Random Forest Feature Importance Analysis')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_indeterminacy_timeseries(self, timestamps: np.ndarray, 
                                    energy_generation: np.ndarray,
                                    indeterminacy: np.ndarray,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot indeterminacy component over time with energy generation.
        
        Args:
            timestamps: Time stamps
            energy_generation: Energy generation values
            indeterminacy: Indeterminacy values
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot energy generation
        ax1.plot(timestamps, energy_generation, color='blue', alpha=0.7, linewidth=1)
        ax1.set_ylabel('Energy Generation (MW)')
        ax1.set_title('Energy Generation and Neutrosophic Indeterminacy Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot indeterminacy
        ax2.plot(timestamps, indeterminacy, color='red', alpha=0.8, linewidth=1.5)
        ax2.set_ylabel('Indeterminacy (I)')
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        
        # Highlight high indeterminacy periods
        high_indeterminacy = indeterminacy > np.percentile(indeterminacy, 90)
        ax2.fill_between(timestamps, 0, indeterminacy, where=high_indeterminacy, 
                        color='red', alpha=0.3, label='High Indeterminacy')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_intervals(self, timestamps: np.ndarray,
                                true_values: np.ndarray,
                                predictions: np.ndarray,
                                lower_bounds: np.ndarray,
                                upper_bounds: np.ndarray,
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot prediction intervals with true values and predictions.
        
        Args:
            timestamps: Time stamps
            true_values: True energy generation values
            predictions: Predicted values
            lower_bounds: Lower bounds of prediction intervals
            upper_bounds: Upper bounds of prediction intervals
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot true values
        ax.plot(timestamps, true_values, color='black', linewidth=2, label='True Values', alpha=0.8)
        
        # Plot predictions
        ax.plot(timestamps, predictions, color='blue', linewidth=1.5, label='Predictions', alpha=0.8)
        
        # Plot prediction intervals
        ax.fill_between(timestamps, lower_bounds, upper_bounds, 
                       color='blue', alpha=0.2, label='95% Prediction Interval')
        
        # Calculate and display coverage
        coverage = np.mean((true_values >= lower_bounds) & (true_values <= upper_bounds))
        ax.text(0.02, 0.98, f'Coverage: {coverage:.1%}', transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy Generation (MW)')
        ax.set_title('Prediction Intervals with Neutrosophic Uncertainty Quantification')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_tsne_features(self, features_vanilla: np.ndarray, 
                          features_neutrosophic: np.ndarray,
                          labels: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
        """Plot t-SNE visualization comparing feature spaces.
        
        Args:
            features_vanilla: Features from vanilla Random Forest
            features_neutrosophic: Features from neutrosophic Random Forest
            labels: Optional labels for coloring points
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not TSNE_AVAILABLE:
            logger.warning("t-SNE visualization requires scikit-learn")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        
        # Vanilla features
        tsne_vanilla = tsne.fit_transform(features_vanilla)
        scatter1 = ax1.scatter(tsne_vanilla[:, 0], tsne_vanilla[:, 1], 
                              c=labels if labels is not None else 'blue', 
                              alpha=0.6, s=20)
        ax1.set_title('Vanilla Random Forest Feature Space')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        
        # Neutrosophic features
        tsne_neutrosophic = tsne.fit_transform(features_neutrosophic)
        scatter2 = ax2.scatter(tsne_neutrosophic[:, 0], tsne_neutrosophic[:, 1], 
                              c=labels if labels is not None else 'red', 
                              alpha=0.6, s=20)
        ax2.set_title('Neutrosophic Random Forest Feature Space')
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_sensitivity_analysis(self, sensitivity_results: Dict[str, Dict],
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot sensitivity analysis results.

        Args:
            sensitivity_results: Dictionary with sensitivity analysis results
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        n_params = len(sensitivity_results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (param_name, param_results) in enumerate(sensitivity_results.items()):
            if i >= len(axes):
                break

            # Extract parameter values and RMSE scores
            param_values = []
            rmse_values = []

            for param_val, rmse in param_results.items():
                if rmse is not None:
                    param_values.append(param_val)
                    rmse_values.append(rmse)

            if param_values:
                # Sort by parameter value
                sorted_data = sorted(zip(param_values, rmse_values))
                param_values, rmse_values = zip(*sorted_data)

                # Plot
                axes[i].plot(param_values, rmse_values, 'o-', linewidth=2, markersize=8)
                axes[i].set_xlabel(param_name.replace('_', ' ').title())
                axes[i].set_ylabel('RMSE')
                axes[i].set_title(f'Sensitivity to {param_name.replace("_", " ").title()}')
                axes[i].grid(True, alpha=0.3)

                # Highlight optimal value
                min_idx = np.argmin(rmse_values)
                axes[i].scatter(param_values[min_idx], rmse_values[min_idx],
                              color='red', s=100, zorder=5, label='Optimal')
                axes[i].legend()

        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_computational_analysis(self, computational_results: Dict[str, Any],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Plot computational complexity analysis.

        Args:
            computational_results: Computational analysis results
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Training time analysis
        if 'training_times' in computational_results:
            for model_name in ['NDC-RF', 'N-BEATS', 'LSTM', 'Transformer']:
                sizes = []
                times = []

                for size, model_times in computational_results['training_times'].items():
                    if model_name in model_times:
                        sizes.append(size)
                        times.append(model_times[model_name])

                if sizes:
                    ax1.plot(sizes, times, 'o-', label=model_name, linewidth=2, markersize=8)

            ax1.set_xlabel('Dataset Size')
            ax1.set_ylabel('Training Time (seconds)')
            ax1.set_title('Training Time Scalability')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Prediction time analysis
        if 'prediction_times' in computational_results:
            for model_name in ['NDC-RF', 'N-BEATS', 'LSTM', 'Transformer']:
                sizes = []
                times = []

                for size, model_times in computational_results['prediction_times'].items():
                    if model_name in model_times:
                        sizes.append(size)
                        times.append(model_times[model_name])

                if sizes:
                    ax2.plot(sizes, times, 'o-', label=model_name, linewidth=2, markersize=8)

            ax2.set_xlabel('Dataset Size')
            ax2.set_ylabel('Prediction Time (seconds)')
            ax2.set_title('Prediction Time Scalability')
            ax2.set_xscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_ablation_study(self, ablation_results: Dict[str, Any],
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot ablation study results.

        Args:
            ablation_results: Ablation study results
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Extract results
        config_names = []
        rmse_values = []
        picp_values = []

        for config_name, results in ablation_results.items():
            if 'point_metrics' in results and not results.get('failed', False):
                config_names.append(config_name.replace('_', ' ').title())
                rmse_values.append(results['point_metrics']['rmse'])
                if 'interval_metrics' in results:
                    picp_values.append(results['interval_metrics']['picp'])
                else:
                    picp_values.append(0)

        # RMSE comparison
        bars1 = ax1.bar(range(len(config_names)), rmse_values)

        # Highlight full model
        for i, name in enumerate(config_names):
            if 'full' in name.lower():
                bars1[i].set_color('green')
                bars1[i].set_alpha(0.8)

        ax1.set_xticks(range(len(config_names)))
        ax1.set_xticklabels(config_names, rotation=45, ha='right')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Ablation Study: Point Forecast Accuracy')
        ax1.grid(True, alpha=0.3)

        # PICP comparison
        if any(picp > 0 for picp in picp_values):
            bars2 = ax2.bar(range(len(config_names)), picp_values)

            # Highlight full model
            for i, name in enumerate(config_names):
                if 'full' in name.lower():
                    bars2[i].set_color('green')
                    bars2[i].set_alpha(0.8)

            ax2.set_xticks(range(len(config_names)))
            ax2.set_xticklabels(config_names, rotation=45, ha='right')
            ax2.set_ylabel('PICP')
            ax2.set_title('Ablation Study: Prediction Interval Coverage')
            ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target Coverage')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_robustness_analysis(self, robustness_results: Dict[str, Any],
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot robustness analysis results.

        Args:
            robustness_results: Robustness analysis results
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Noise robustness
        if 'noise_robustness' in robustness_results:
            noise_levels = []
            rmse_values = []

            for noise_level, rmse in robustness_results['noise_robustness'].items():
                if rmse is not None:
                    noise_levels.append(noise_level * 100)  # Convert to percentage
                    rmse_values.append(rmse)

            if noise_levels:
                ax1.plot(noise_levels, rmse_values, 'o-', linewidth=2, markersize=8, color='red')
                ax1.set_xlabel('Noise Level (%)')
                ax1.set_ylabel('RMSE')
                ax1.set_title('Robustness to Noise')
                ax1.grid(True, alpha=0.3)

        # Missing data robustness
        if 'missing_data_robustness' in robustness_results:
            missing_rates = []
            rmse_values = []

            for missing_rate, rmse in robustness_results['missing_data_robustness'].items():
                if rmse is not None:
                    missing_rates.append(missing_rate * 100)  # Convert to percentage
                    rmse_values.append(rmse)

            if missing_rates:
                ax2.plot(missing_rates, rmse_values, 'o-', linewidth=2, markersize=8, color='blue')
                ax2.set_xlabel('Missing Data Rate (%)')
                ax2.set_ylabel('RMSE')
                ax2.set_title('Robustness to Missing Data')
                ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def generate_all_plots(self, results: Dict[str, Any], output_dir: Optional[str] = None):
        """Generate all experimental plots from comprehensive results.

        Args:
            results: Complete experimental results
            output_dir: Directory to save plots
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating all experimental plots")

        # Main results comparison
        if 'main_results' in results:
            fig = self.plot_model_comparison(results['main_results'])
            fig.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        # Statistical significance heatmap
        if 'main_results' in results:
            statistical_data = {}
            for dataset, dataset_results in results['main_results'].items():
                if 'statistical_tests' in dataset_results:
                    statistical_data[dataset] = dataset_results['statistical_tests']

            if statistical_data:
                fig = self.plot_statistical_significance_heatmap(statistical_data)
                fig.savefig(self.output_dir / 'statistical_significance.png', dpi=300, bbox_inches='tight')
                plt.close(fig)

        # Ablation study
        if 'ablation_studies' in results:
            fig = self.plot_ablation_study(results['ablation_studies'])
            fig.savefig(self.output_dir / 'ablation_study.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        # Sensitivity analysis
        if 'sensitivity_analysis' in results:
            fig = self.plot_sensitivity_analysis(results['sensitivity_analysis'])
            fig.savefig(self.output_dir / 'sensitivity_analysis.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        # Computational analysis
        if 'computational_analysis' in results:
            fig = self.plot_computational_analysis(results['computational_analysis'])
            fig.savefig(self.output_dir / 'computational_analysis.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        # Robustness analysis
        if 'robustness_analysis' in results:
            fig = self.plot_robustness_analysis(results['robustness_analysis'])
            fig.savefig(self.output_dir / 'robustness_analysis.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        logger.info(f"All plots saved to {self.output_dir}")
