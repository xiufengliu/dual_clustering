"""
Data Exploration Notebook for Renewable Energy Forecasting

This script demonstrates data loading, exploration, and visualization
for the neutrosophic forecasting framework.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import ENTSOEDataLoader
from src.data.validator import DataValidator
from src.data.preprocessor import DataPreprocessor
from src.utils.visualization import plot_time_series, plot_cluster_analysis
from src.utils.logger import setup_logger

# Setup
logger = setup_logger("data_exploration")
plt.style.use('seaborn-v0_8')


def load_and_explore_data():
    """Load and explore renewable energy data."""
    print("="*60)
    print("RENEWABLE ENERGY DATA EXPLORATION")
    print("="*60)
    
    # Initialize data loader
    data_loader = ENTSOEDataLoader()
    
    # Load solar data
    print("\n1. Loading Solar Data...")
    solar_data = data_loader.load_solar_data(
        start_date="2019-01-01",
        end_date="2023-10-03",
        country="Denmark"
    )
    
    print(f"Solar data shape: {solar_data.shape}")
    print(f"Date range: {solar_data['timestamp'].min()} to {solar_data['timestamp'].max()}")
    
    # Load wind data
    print("\n2. Loading Wind Data...")
    wind_data = data_loader.load_wind_data(
        start_date="2019-01-01", 
        end_date="2023-10-03",
        country="Denmark"
    )
    
    print(f"Wind data shape: {wind_data.shape}")
    print(f"Date range: {wind_data['timestamp'].min()} to {wind_data['timestamp'].max()}")
    
    return solar_data, wind_data


def validate_data(data, data_type):
    """Validate data quality."""
    print(f"\n3. Validating {data_type} Data...")
    
    validator = DataValidator()
    is_valid, report = validator.validate_dataset(data)
    
    print(f"Data valid: {is_valid}")
    if report['errors']:
        print(f"Errors: {report['errors']}")
    if report['warnings']:
        print(f"Warnings: {report['warnings']}")
    
    # Print statistics
    stats = report['statistics']
    print(f"\nData Statistics:")
    print(f"  Total points: {stats['total_points']}")
    print(f"  Valid points: {stats['valid_points']}")
    print(f"  Missing ratio: {stats['missing_ratio']:.4f}")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Min: {stats['min']:.2f}")
    print(f"  Max: {stats['max']:.2f}")


def explore_data_patterns(data, data_type):
    """Explore patterns in the data."""
    print(f"\n4. Exploring {data_type} Data Patterns...")
    
    # Basic statistics
    energy_values = data['energy_generation'].values
    
    print(f"Energy Generation Statistics:")
    print(f"  Mean: {np.mean(energy_values):.2f} MW")
    print(f"  Median: {np.median(energy_values):.2f} MW")
    print(f"  Std: {np.std(energy_values):.2f} MW")
    print(f"  Min: {np.min(energy_values):.2f} MW")
    print(f"  Max: {np.max(energy_values):.2f} MW")
    print(f"  Zero values: {np.sum(energy_values == 0)}")
    
    # Time-based analysis
    data_with_time = data.copy()
    data_with_time['hour'] = data_with_time['timestamp'].dt.hour
    data_with_time['day_of_week'] = data_with_time['timestamp'].dt.dayofweek
    data_with_time['month'] = data_with_time['timestamp'].dt.month
    
    # Hourly patterns
    hourly_mean = data_with_time.groupby('hour')['energy_generation'].mean()
    print(f"\nHourly Pattern (peak hour: {hourly_mean.idxmax()})")
    print(f"  Peak generation: {hourly_mean.max():.2f} MW")
    print(f"  Minimum generation: {hourly_mean.min():.2f} MW")
    
    # Monthly patterns
    monthly_mean = data_with_time.groupby('month')['energy_generation'].mean()
    print(f"\nMonthly Pattern (peak month: {monthly_mean.idxmax()})")
    print(f"  Peak generation: {monthly_mean.max():.2f} MW")
    print(f"  Minimum generation: {monthly_mean.min():.2f} MW")


def visualize_data(solar_data, wind_data):
    """Create visualizations of the data."""
    print("\n5. Creating Visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Solar time series
    axes[0, 0].plot(solar_data['timestamp'], solar_data['energy_generation'], 
                   alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title('Solar Power Generation Time Series')
    axes[0, 0].set_ylabel('Generation (MW)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Wind time series
    axes[0, 1].plot(wind_data['timestamp'], wind_data['energy_generation'], 
                   alpha=0.7, linewidth=0.5, color='orange')
    axes[0, 1].set_title('Wind Power Generation Time Series')
    axes[0, 1].set_ylabel('Generation (MW)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Solar distribution
    axes[1, 0].hist(solar_data['energy_generation'], bins=50, alpha=0.7, 
                   edgecolor='black')
    axes[1, 0].set_title('Solar Power Generation Distribution')
    axes[1, 0].set_xlabel('Generation (MW)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Wind distribution
    axes[1, 1].hist(wind_data['energy_generation'], bins=50, alpha=0.7, 
                   color='orange', edgecolor='black')
    axes[1, 1].set_title('Wind Power Generation Distribution')
    axes[1, 1].set_xlabel('Generation (MW)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Seasonal patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Add time features
    for data, name, ax_row in [(solar_data, 'Solar', 0), (wind_data, 'Wind', 1)]:
        data_temp = data.copy()
        data_temp['hour'] = data_temp['timestamp'].dt.hour
        data_temp['month'] = data_temp['timestamp'].dt.month
        
        # Hourly pattern
        hourly_pattern = data_temp.groupby('hour')['energy_generation'].mean()
        axes[ax_row, 0].plot(hourly_pattern.index, hourly_pattern.values, 'o-')
        axes[ax_row, 0].set_title(f'{name} Power - Hourly Pattern')
        axes[ax_row, 0].set_xlabel('Hour of Day')
        axes[ax_row, 0].set_ylabel('Average Generation (MW)')
        axes[ax_row, 0].grid(True, alpha=0.3)
        
        # Monthly pattern
        monthly_pattern = data_temp.groupby('month')['energy_generation'].mean()
        axes[ax_row, 1].plot(monthly_pattern.index, monthly_pattern.values, 's-')
        axes[ax_row, 1].set_title(f'{name} Power - Monthly Pattern')
        axes[ax_row, 1].set_xlabel('Month')
        axes[ax_row, 1].set_ylabel('Average Generation (MW)')
        axes[ax_row, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/seasonal_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()


def preprocess_and_analyze(data, data_type):
    """Demonstrate preprocessing and analysis."""
    print(f"\n6. Preprocessing {data_type} Data...")
    
    preprocessor = DataPreprocessor()
    
    # Preprocess data
    normalized_data, params = preprocessor.preprocess(data, fit=True)
    
    print(f"Original range: [{np.min(data['energy_generation']):.2f}, {np.max(data['energy_generation']):.2f}]")
    print(f"Normalized range: [{np.min(normalized_data):.4f}, {np.max(normalized_data):.4f}]")
    
    # Get statistics
    stats = preprocessor.get_data_statistics(data)
    print(f"Data statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return normalized_data, params


def main():
    """Main exploration function."""
    # Create output directories
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        solar_data, wind_data = load_and_explore_data()
        
        # Validate data
        validate_data(solar_data, "Solar")
        validate_data(wind_data, "Wind")
        
        # Explore patterns
        explore_data_patterns(solar_data, "Solar")
        explore_data_patterns(wind_data, "Wind")
        
        # Create visualizations
        visualize_data(solar_data, wind_data)
        
        # Preprocessing analysis
        solar_normalized, solar_params = preprocess_and_analyze(solar_data, "Solar")
        wind_normalized, wind_params = preprocess_and_analyze(wind_data, "Wind")
        
        print("\n" + "="*60)
        print("DATA EXPLORATION COMPLETED")
        print("="*60)
        print("Visualizations saved to results/figures/")
        
    except Exception as e:
        logger.error(f"Data exploration failed: {str(e)}", exc_info=True)
        print(f"✗ Data exploration failed: {str(e)}")


if __name__ == "__main__":
    main()