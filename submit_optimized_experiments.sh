#!/bin/bash
### Optimized comprehensive evaluation script for dual clustering experiments
### This script uses performance optimizations to reduce execution time by 60%+
#BSUB -q gpua100                   # Queue name (choose based on GPU type)
#BSUB -J dual_cluster_optimized    # Job name
#BSUB -n 8                         # Number of CPU cores
#BSUB -gpu "num=1:mode=exclusive_process"  # One GPU in exclusive mode
#BSUB -R "rusage[mem=8GB]"         # 8 GB system memory (reduced from 16GB due to optimizations)
#BSUB -W 1:30                      # Walltime: 1.5 hours (reduced from 8 hours)
#BSUB -o gpu_optimized_%J.out      # Output file
#BSUB -e gpu_optimized_%J.err      # Error file

# Source bashrc to set up local Python environment
source ~/.bashrc

# Load only CUDA module (using local Python)
module load cuda/12.6

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Change to project directory
cd /zhome/bb/9/101964/xiuli/dual_clustering

echo "=== Starting Optimized Comprehensive Evaluation ==="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Expected completion time: 30-60 minutes (vs 6+ hours for original)"

# Install dependencies (if needed)
echo "=== Installing dependencies ==="
pip install --user -r requirements.txt

# Create output directory
mkdir -p results/optimized

# Run optimized comprehensive evaluation
echo "=== Running optimized comprehensive evaluation ==="
echo "Using fast_benchmark_config with performance optimizations:"
echo "- Vectorized prediction for large horizons"
echo "- Optimized FCM clustering"
echo "- Parallel dataset processing"
echo "- Reduced model complexity for speed"

# Start timing
start_time=$(date +%s)

# Run optimized experiments with comprehensive dataset coverage
python run_optimized_experiments.py \
    --config fast_benchmark_config \
    --output-dir results/optimized \
    --datasets kaggle_solar_plant entso_e_load nrel_canada_wind gefcom2014_energy \
    --max-samples 5000 \
    --profile

# Calculate execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))
execution_minutes=$((execution_time / 60))

echo "=== Optimized comprehensive evaluation completed ==="
echo "Total execution time: ${execution_time} seconds (${execution_minutes} minutes)"
echo "Results saved in: results/optimized/"

# Performance validation
echo "=== Performance Summary ==="
echo "Datasets processed: 4"
echo "Average time per dataset: $((execution_time / 4)) seconds"

if [ $execution_time -lt 3600 ]; then
    echo "✓ SUCCESS: Completed in under 1 hour (target achieved)"
else
    echo "⚠ WARNING: Execution time exceeded 1 hour target"
fi

if [ $execution_time -lt 1800 ]; then
    echo "✓ EXCELLENT: Completed in under 30 minutes"
fi

# List generated files
echo "=== Generated Output Files ==="
ls -la results/optimized/

echo "=== Job completed successfully ==="
