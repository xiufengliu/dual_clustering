#!/bin/bash
### Comprehensive evaluation script for dual clustering experiments
#BSUB -q gpua100                   # Queue name (choose based on GPU type)
#BSUB -J dual_cluster_comprehensive # Job name
#BSUB -n 8                         # Number of CPU cores
#BSUB -gpu "num=1:mode=exclusive_process"  # One GPU in exclusive mode
#BSUB -R "rusage[mem=16GB]"        # 16 GB system memory
#BSUB -W 8:00                      # Walltime: 8 hours
#BSUB -o gpu_comprehensive_%J.out  # Output file
#BSUB -e gpu_comprehensive_%J.err  # Error file

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

echo "=== Starting Comprehensive Evaluation ==="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Install dependencies
echo "=== Installing dependencies ==="
pip install --user -r requirements.txt

# Run comprehensive evaluation
echo "=== Running comprehensive evaluation ==="

# Run full comprehensive evaluation with all components
python experiments/comprehensive_evaluation.py --config benchmark_config

# Alternative: Run specific components only (uncomment if needed)
# python experiments/comprehensive_evaluation.py \
#     --config benchmark_config \
#     --datasets entso_e_solar entso_e_wind gefcom2014_solar \
#     --skip-computational --skip-cross-dataset --skip-robustness

echo "=== Comprehensive evaluation completed ==="
echo "Results saved in: results/comprehensive/"
