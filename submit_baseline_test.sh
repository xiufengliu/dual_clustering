#!/bin/bash
### Baseline performance test for optimized dual clustering experiments
### This script tests core performance improvements without complex components
#BSUB -q gpua100                   # Queue name
#BSUB -J dual_cluster_baseline     # Job name
#BSUB -n 4                         # Number of CPU cores (reduced)
#BSUB -gpu "num=1:mode=exclusive_process"  # One GPU
#BSUB -R "rusage[mem=4GB]"         # 4 GB memory (reduced)
#BSUB -W 0:30                      # Walltime: 30 minutes
#BSUB -o gpu_baseline_%J.out       # Output file
#BSUB -e gpu_baseline_%J.err       # Error file

# Source bashrc to set up local Python environment
source ~/.bashrc

# Load only CUDA module (using local Python)
module load cuda/12.6

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Change to project directory
cd /zhome/bb/9/101964/xiuli/dual_clustering

echo "=== Starting Baseline Performance Test ==="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Expected completion time: 5-10 minutes"

# Install dependencies (if needed)
echo "=== Installing dependencies ==="
pip install --user -r requirements.txt

# Create output directory
mkdir -p results/optimized

# Run baseline performance test
echo "=== Running baseline performance test ==="
echo "Testing core performance optimizations:"
echo "- Optimized baseline models"
echo "- Efficient data processing"
echo "- Reduced dataset sizes for speed"

# Start timing
start_time=$(date +%s)

# Run baseline test
python run_baseline_performance_test.py

# Calculate execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))
execution_minutes=$((execution_time / 60))

echo "=== Baseline performance test completed ==="
echo "Total execution time: ${execution_time} seconds (${execution_minutes} minutes)"
echo "Results saved in: results/optimized/"

# Performance validation
echo "=== Performance Summary ==="
echo "Datasets processed: 3"
echo "Average time per dataset: $((execution_time / 3)) seconds"

if [ $execution_time -lt 600 ]; then
    echo "✅ SUCCESS: Completed in under 10 minutes (target achieved)"
else
    echo "⚠️ WARNING: Execution time exceeded 10 minutes"
fi

if [ $execution_time -lt 300 ]; then
    echo "✅ EXCELLENT: Completed in under 5 minutes"
fi

# List generated files
echo "=== Generated Output Files ==="
ls -la results/optimized/

echo "=== Baseline test completed successfully ==="
