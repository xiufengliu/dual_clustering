# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of a **Neutrosophic Dual Clustering Random Forest Framework** for renewable energy forecasting. The framework combines dual clustering (K-means + Fuzzy C-means), neutrosophic logic transformation, and Random Forest prediction to achieve uncertainty-aware energy forecasting.

## Key Architecture Components

### Core Pipeline (src/framework/forecasting_framework.py:22-463)
The main `NeutrosophicForecastingFramework` class implements the complete Algorithm 1 pipeline:
1. **Data preprocessing** - Normalization and feature engineering
2. **Dual clustering** - K-means and FCM clustering in parallel
3. **Neutrosophic transformation** - Converts clustering outputs to Truth/Indeterminacy/Falsity components
4. **Random Forest training** - Uses enriched features for prediction
5. **Prediction with intervals** - Uncertainty quantification using neutrosophic indeterminacy

### Clustering Module (src/clustering/)
- `DualClusterer` (dual_clusterer.py:13-391) - Orchestrates K-means and FCM clustering
- `KMeansClusterer` - Standard K-means implementation
- `FCMClusterer` - Fuzzy C-means implementation
- Creates integrated features: `[one_hot_kmeans, fcm_memberships]`

### Neutrosophic Module (src/neutrosophic/)
- `NeutrosophicTransformer` (neutrosophic_transformer.py:39-385) - Core transformation logic
- Implements Definition 3 from paper: T(yi) = ui,ki, F(yi) = 1-T(yi), I(yi) = H(ui)/log₂(C)
- `NeutrosophicComponents` dataclass for T/I/F storage
- `UncertaintyQuantifier` for uncertainty analysis

### Models Module (src/models/)
- `RandomForestForecaster` - Enhanced Random Forest with neutrosophic interval prediction
- `BaselineForecasters` - 12 baseline models (ARIMA, LSTM, Transformer, etc.)
- `EnsemblePredictor` - Model combination utilities

### Data Module (src/data/)
- `ENTSOEDataLoader` - Handles 15 real-world renewable energy datasets
- `DataPreprocessor` - Normalization, feature engineering, lag creation
- `DataValidator` - Data quality checks and validation

## Common Development Commands

### Installation & Setup
```bash
# Install with development dependencies
make install-dev
# or
pip install -e ".[dev,notebooks,experiments]"

# Install basic dependencies only
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
make test
# or
pytest tests/ -v --cov=src --cov-report=html

# Run fast tests (skip slow ones)
make test-fast
# or
pytest tests/ -v -x --disable-warnings

# Test specific modules
pytest tests/test_clustering.py
pytest tests/test_neutrosophic.py
pytest tests/test_framework.py
```

### Code Quality
```bash
# Format code
make format
# or
black src tests scripts experiments notebooks
isort src tests scripts experiments notebooks

# Run linting
make lint
# or
flake8 src tests scripts experiments
mypy src

# Check formatting without changes
make format-check
```

### Training & Evaluation
```bash
# Train individual models
make train-solar    # Solar energy model
make train-wind     # Wind energy model

# Evaluate trained models
make evaluate

# Run experiments
make experiment                 # Basic experiment
make experiment-ablation       # Ablation studies
make experiment-multiple       # Multiple configurations
```

### Comprehensive Experiments
```bash
# Local testing first
chmod +x test_local.sh && ./test_local.sh

# Quick comprehensive evaluation
python run_complete_experiments.py --step quick

# Main comparison experiments
python run_complete_experiments.py --step main

# Full experimental suite
python run_complete_experiments.py --step full

# Generate visualizations
python run_complete_experiments.py --step viz
```

### HPC/GPU Cluster Usage
```bash
# Submit comprehensive evaluation to GPU cluster
chmod +x submit_comprehensive_eval.sh
bsub < submit_comprehensive_eval.sh

# Monitor job progress
bjobs                                    # Check job status
tail -f gpu_comprehensive_*.out         # Watch output
watch -n 30 'tail -20 gpu_comprehensive_*.out'  # Real-time monitoring
```

## Key Configuration Files

### Main Configs (config/)
- `base_config.yaml` - Default framework configuration
- `solar_config.yaml` - Solar-specific parameters
- `wind_config.yaml` - Wind-specific parameters
- `experiment_configs/benchmark_config.yaml` - Comprehensive evaluation settings

### Important Parameters
- `clustering.n_clusters: 5` - Number of clusters for both K-means and FCM
- `clustering.fcm_fuzziness: 2.0` - FCM fuzziness parameter
- `neutrosophic.entropy_epsilon: 1e-9` - Numerical stability for entropy calculation
- `random_forest.n_estimators: 100` - Number of trees in Random Forest

## Data Structure

### Available Datasets (data/processed/)
The repository includes 15 real-world renewable energy datasets:
- **Individual datasets**: GEFCom2014, Kaggle, NREL, UK Sheffield, ENTSO-E (6 sources)
- **Combined datasets**: Combined solar/wind/load data (3 unified collections)
- **Legacy synthetic datasets**: Backward compatibility (6 datasets)

### Data Format
All datasets follow standardized format:
- `timestamp` - DateTime index
- `energy_generation` - Target variable (MW)
- Time features: `hour`, `day_of_week`, `month`, etc.
- Weather features: `temperature`, `wind_speed`, `cloud_cover`, etc.

## Experimental Framework

### Baseline Models (12 total)
- Statistical: ARIMA, SARIMA
- Machine Learning: SVR, LightGBM, MLP, Random Forest
- Deep Learning: LSTM, CNN-LSTM, N-BEATS, Transformer, Informer, TiDE

### Evaluation Metrics
- **Point forecasting**: RMSE, MAE, MAPE, sMAPE, R², MBE
- **Interval forecasting**: PICP, PINAW, Winkler Score, CWC
- **Statistical tests**: Modified Diebold-Mariano, Friedman test

### Experiment Components
1. **Main comparison** - NDC-RF vs 12 baselines
2. **Ablation studies** - Component contribution analysis
3. **Sensitivity analysis** - Hyperparameter robustness
4. **Computational analysis** - Time/memory profiling
5. **Cross-dataset generalization** - Domain adaptation
6. **Robustness analysis** - Noise/missing data handling

## Development Workflow

### Adding New Features
1. Implement in appropriate module (src/clustering/, src/neutrosophic/, etc.)
2. Add unit tests in tests/
3. Update configuration if needed
4. Run formatting and linting
5. Test with quick experiments

### Debugging Common Issues
- **Memory errors**: Reduce `n_clusters` or dataset size in config
- **Convergence issues**: Increase `max_iter` or `tol` in clustering config
- **Import errors**: Check PYTHONPATH includes project root
- **Data validation failures**: Check data format and missing values

### Performance Optimization
- Use `--parallel` flag for multi-core processing
- Reduce `n_runs` in config for faster testing
- Skip expensive components with `--skip-*` flags
- Use GPU acceleration for deep learning baselines

## Results & Reproducibility

### Output Structure
```
results/
├── comprehensive/          # Main experiment results (JSON)
├── figures/               # Generated visualizations
├── logs/                  # Execution logs
└── models/               # Trained model files
```

### Reproducibility Features
- Fixed random seeds across all components
- Comprehensive logging with timestamps
- Statistical significance testing
- Multiple runs with averaging (default: 5 runs)

## Quick Start Commands

```bash
# Complete setup and test
make install-dev && make test

# Quick local test
./test_local.sh

# Run quick experiments
python run_complete_experiments.py --step quick

# Submit to cluster
bsub < submit_comprehensive_eval.sh
```