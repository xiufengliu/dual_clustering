"""Visualization utilities for the neutrosophic forecasting framework."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_time_series(data: np.ndarray, timestamps: Optional[np.ndarray] = None,
                    title: str = "Time Series", xlabel: str = "Time", 
                    ylabel: str = "Value", figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """Plot time series data.
    
    Args:
        data: Time series values
        timestamps: Optional timestamp array
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x_values = timestamps if timestamps is not None else np.arange(len(data))
    ax.plot(x_values, data, linewidth=1.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_predictions_with_intervals(actual: np.ndarray, predictions: np.ndarray,
                                  lower_bounds: np.ndarray, upper_bounds: np.ndarray,
                                  timestamps: Optional[np.ndarray] = None,
                                  title: str = "Predictions with Confidence Intervals",
                                  figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
    """Plot predictions with confidence intervals.
    
    Args:
        actual: Actual values
        predictions: Predicted values
        lower_bounds: Lower confidence bounds
        upper_bounds: Upper confidence bounds
        timestamps: Optional timestamp array
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x_values = timestamps if timestamps is not None else np.arange(len(actual))
    
    # Plot actual values
    ax.plot(x_values, actual, label='Actual', color='black', linewidth=2)
    
    # Plot predictions
    ax.plot(x_values, predictions, label='Predicted', color='red', linewidth=2)
    
    # Plot confidence intervals
    ax.fill_between(x_values, lower_bounds, upper_bounds, 
                   alpha=0.3, color='red', label='Confidence Interval')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Energy Generation (MW)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_cluster_analysis(data: np.ndarray, kmeans_labels: np.ndarray,
                         fcm_memberships: np.ndarray, cluster_centers: np.ndarray,
                         title: str = "Cluster Analysis", 
                         figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """Plot cluster analysis results.
    
    Args:
        data: Original data points
        kmeans_labels: K-means cluster labels
        fcm_memberships: FCM membership matrix
        cluster_centers: Cluster centers
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Original time series with K-means clusters
    axes[0, 0].scatter(np.arange(len(data)), data, c=kmeans_labels, 
                      cmap='viridis', alpha=0.6, s=20)
    axes[0, 0].set_title('K-Means Clustering', fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Energy Generation')
    
    # Plot 2: FCM membership heatmap
    im = axes[0, 1].imshow(fcm_memberships.T, aspect='auto', cmap='viridis')
    axes[0, 1].set_title('FCM Membership Matrix', fontweight='bold')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Cluster')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Plot 3: Cluster centers
    axes[1, 0].bar(range(len(cluster_centers)), cluster_centers)
    axes[1, 0].set_title('Cluster Centers', fontweight='bold')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Center Value')
    
    # Plot 4: Membership distribution
    for i in range(fcm_memberships.shape[1]):
        axes[1, 1].hist(fcm_memberships[:, i], alpha=0.5, 
                       label=f'Cluster {i}', bins=30)
    axes[1, 1].set_title('FCM Membership Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Membership Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_neutrosophic_components(truth: np.ndarray, indeterminacy: np.ndarray,
                                falsity: np.ndarray, timestamps: Optional[np.ndarray] = None,
                                title: str = "Neutrosophic Components",
                                figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """Plot neutrosophic components (T, I, F).
    
    Args:
        truth: Truth membership values
        indeterminacy: Indeterminacy values
        falsity: Falsity membership values
        timestamps: Optional timestamp array
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    x_values = timestamps if timestamps is not None else np.arange(len(truth))
    
    # Plot Truth component
    axes[0, 0].plot(x_values, truth, color='green', linewidth=1.5)
    axes[0, 0].set_title('Truth (T)', fontweight='bold')
    axes[0, 0].set_ylabel('Truth Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot Indeterminacy component
    axes[0, 1].plot(x_values, indeterminacy, color='orange', linewidth=1.5)
    axes[0, 1].set_title('Indeterminacy (I)', fontweight='bold')
    axes[0, 1].set_ylabel('Indeterminacy Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot Falsity component
    axes[1, 0].plot(x_values, falsity, color='red', linewidth=1.5)
    axes[1, 0].set_title('Falsity (F)', fontweight='bold')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Falsity Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot all components together
    axes[1, 1].plot(x_values, truth, label='Truth', color='green', linewidth=1.5)
    axes[1, 1].plot(x_values, indeterminacy, label='Indeterminacy', color='orange', linewidth=1.5)
    axes[1, 1].plot(x_values, falsity, label='Falsity', color='red', linewidth=1.5)
    axes[1, 1].set_title('All Components', fontweight='bold')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Component Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_evaluation_metrics(metrics_dict: Dict[str, float], 
                           title: str = "Evaluation Metrics",
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot evaluation metrics as bar chart.
    
    Args:
        metrics_dict: Dictionary of metric names and values
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    bars = ax.bar(metrics, values, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12)
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names: List[str], importances: np.ndarray,
                           title: str = "Feature Importance",
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """Plot feature importance from Random Forest.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    bars = ax.barh(range(len(sorted_features)), sorted_importances, 
                   color='lightcoral', edgecolor='darkred', alpha=0.7)
    
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, sorted_importances)):
        width = bar.get_width()
        ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                f'{importance:.4f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_interactive_forecast_plot(actual: np.ndarray, predictions: np.ndarray,
                                   lower_bounds: np.ndarray, upper_bounds: np.ndarray,
                                   timestamps: Optional[np.ndarray] = None,
                                   title: str = "Interactive Forecast Plot") -> go.Figure:
    """Create interactive forecast plot using Plotly.
    
    Args:
        actual: Actual values
        predictions: Predicted values
        lower_bounds: Lower confidence bounds
        upper_bounds: Upper confidence bounds
        timestamps: Optional timestamp array
        title: Plot title
        
    Returns:
        Plotly figure
    """
    x_values = timestamps if timestamps is not None else np.arange(len(actual))
    
    fig = go.Figure()
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_values, x_values[::-1]]),
        y=np.concatenate([upper_bounds, lower_bounds[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True
    ))
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=x_values,
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='black', width=2)
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=x_values,
        y=predictions,
        mode='lines',
        name='Predicted',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Energy Generation (MW)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig