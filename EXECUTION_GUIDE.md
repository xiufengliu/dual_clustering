# Execution Guide for Dual Clustering Experiments

## Overview

This guide provides step-by-step instructions for running the comprehensive neutrosophic dual clustering experiments on your GPU cluster.

## Prerequisites

1. **Environment**: GPU cluster with CUDA support (CUDA 12.6)
2. **Python**: Local Python 3.8+ installation with pip
3. **Resources**: At least 16GB RAM, 1 GPU, 8 CPU cores

## Step 1: Local Testing (Recommended)

Before submitting to the cluster, test your setup locally:

```bash
# Make the test script executable
chmod +x test_local.sh

# Run local test
./test_local.sh
```

This will:
- Check your Python installation
- Install required dependencies
- Test critical imports
- Verify the setup is ready

## Step 2: Submit Comprehensive Experiment

Run the full comprehensive evaluation (8 hours):

```bash
# Make the script executable
chmod +x submit_comprehensive_eval.sh

# Submit to GPU cluster
bsub < submit_comprehensive_eval.sh
```

This comprehensive experiment includes:
- **Main comparison**: NDC-RF vs 12 baseline models
- **Ablation studies**: Component contribution analysis
- **Sensitivity analysis**: Hyperparameter robustness
- **Statistical testing**: Significance analysis
- **Multiple datasets**: Diverse renewable energy data

## Step 3: Monitor Your Job

```bash
# Check job status
bjobs

# Check job details
bjobs -l <job_id>

# Check output (while running)
tail -f gpu_comprehensive_<job_id>.out

# Check errors
tail -f gpu_comprehensive_<job_id>.err

# Watch progress in real-time
watch -n 30 'tail -20 gpu_comprehensive_*.out'
```

## Step 4: Collect Results

After completion, results will be in:

```
results/
├── comprehensive/
│   ├── comprehensive_evaluation_YYYYMMDD_HHMMSS.json
│   └── comprehensive_evaluation_YYYYMMDD_HHMMSS.log
├── figures/
│   ├── model_comparison.png
│   ├── statistical_significance.png
│   ├── ablation_study.png
│   └── ...
└── logs/
    └── experiment_run_YYYYMMDD_HHMMSS.log
```

## Available Experiment Components

### 1. Main Comparison
- Compares NDC-RF against 12 baseline models
- Statistical significance testing
- Multiple datasets

### 2. Ablation Studies
- Tests individual component contributions
- Without neutrosophic features
- Different clustering approaches

### 3. Sensitivity Analysis
- Hyperparameter robustness
- Number of clusters, fuzziness, etc.

### 4. Computational Analysis
- Training/prediction time
- Memory usage
- Scalability testing

### 5. Cross-Dataset Generalization
- Train on one dataset, test on another
- Domain adaptation capabilities

### 6. Robustness Analysis
- Performance under noise
- Missing data handling

## Customization Options

### Modify Experiment Configuration

Edit `config/experiment_configs/benchmark_config.yaml`:

```yaml
# Reduce runtime for testing
reproducibility:
  n_runs: 1  # Instead of 5

# Skip expensive components
computational_analysis:
  enabled: false

# Reduce dataset sizes
computational_analysis:
  dataset_sizes: [1000, 5000]  # Instead of [1000, 5000, 10000, 20000]
```

### Run Specific Components Only

```bash
# Only main comparison
python experiments/comprehensive_evaluation.py \
    --config benchmark_config \
    --skip-ablation --skip-sensitivity --skip-computational \
    --skip-cross-dataset --skip-robustness

# Only ablation study
python experiments/comprehensive_evaluation.py \
    --config benchmark_config \
    --skip-main --skip-sensitivity --skip-computational \
    --skip-cross-dataset --skip-robustness
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce dataset sizes or use fewer CPU cores
2. **CUDA errors**: Check GPU availability and CUDA version
3. **Import errors**: Ensure all dependencies are installed
4. **Timeout**: Increase walltime in job script

### Performance Optimization

1. **Use parallel processing**: Add `--parallel` flag
2. **Reduce runs**: Set `n_runs: 1` in config for testing
3. **Skip expensive components**: Use `--skip-*` flags

### Debug Mode

For debugging, run interactively:

```bash
# Request interactive session
bsub -Is -q gpua100 -n 4 -gpu "num=1" -R "rusage[mem=8GB]" -W 1:00 bash

# Then run experiments manually
python run_complete_experiments.py --step quick --skip-deps
```

## Expected Runtime

- **Comprehensive Evaluation**: 8 hours (includes all components)
- **Reduced version** (with --skip flags): 2-4 hours

## Next Steps

1. Test your setup locally with `./test_local.sh`
2. Submit comprehensive experiment with `bsub < submit_comprehensive_eval.sh`
3. Monitor job progress with `bjobs` and `tail -f gpu_comprehensive_*.out`
4. Analyze results in `results/` directory after completion
5. Use results for your TNNLS paper submission

## Quick Commands Summary

```bash
# Test setup
chmod +x test_local.sh && ./test_local.sh

# Submit job
chmod +x submit_comprehensive_eval.sh && bsub < submit_comprehensive_eval.sh

# Monitor job
bjobs && tail -f gpu_comprehensive_*.out
```
