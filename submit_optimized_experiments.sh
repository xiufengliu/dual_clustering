#!/bin/bash
#BSUB -J comprehensive_experiments
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 8:00
#BSUB -o gpu_comprehensive_%J.out
#BSUB -e gpu_comprehensive_%J.err
#BSUB -N

# Comprehensive experiment submission script for journal paper publication
# This script runs full comprehensive evaluation with all datasets and components

echo "=== Starting Comprehensive Experiments ==="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Source bashrc to set up local Python environment
source ~/.bashrc

# Load required modules (using local Python as per user preference)
module load cuda/12.6

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

echo "=== Installing dependencies ==="
pip install --user -r requirements.txt

echo "=== Testing bug fixes first ==="
python test_fixes.py
if [ $? -ne 0 ]; then
    echo "❌ Bug fix tests failed! Aborting experiment."
    exit 1
fi
echo "✅ Bug fix tests passed!"

echo "=== Running comprehensive evaluation ==="
# Run full comprehensive evaluation with all components for journal paper
python experiments/comprehensive_evaluation.py --config benchmark_config

echo "=== Comprehensive experiment completed ==="
echo "Results saved to: results/comprehensive/"
echo "Check output files: gpu_comprehensive_${LSB_JOBID}.out and gpu_comprehensive_${LSB_JOBID}.err"