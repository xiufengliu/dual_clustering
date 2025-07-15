#!/bin/bash
#BSUB -J optimized_experiments
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 02:00
#BSUB -o gpu_optimized_%J.out
#BSUB -e gpu_optimized_%J.err
#BSUB -N

# Optimized experiment submission script with bug fixes
# This script runs experiments with reduced computational complexity

echo "=== Starting Optimized Experiments ==="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Load required modules
module load cuda/12.6

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Testing bug fixes first ==="
python test_fixes.py
if [ $? -ne 0 ]; then
    echo "❌ Bug fix tests failed! Aborting experiment."
    exit 1
fi
echo "✅ Bug fix tests passed!"

echo "=== Running optimized experiments ==="
python run_optimized_experiments.py \
    --config fast_benchmark_config \
    --datasets kaggle_solar_plant entso_e_load_fixed nrel_canada_wind \
    --max-samples 2000 \
    --skip-computational \
    --skip-cross-dataset \
    --skip-robustness \
    --output-dir results/optimized > experiment_results.log 2>&1

echo "=== Experiment completed ==="
echo "Results saved to: results/optimized/"
echo "Check output files: gpu_optimized_${LSB_JOBID}.out and gpu_optimized_${LSB_JOBID}.err"