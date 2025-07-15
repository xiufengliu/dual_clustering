#!/bin/bash

# Local testing script for the optimized experiments
# This script runs a minimal test to validate the fixes work in the full pipeline

echo "=========================================="
echo "LOCAL TESTING OF OPTIMIZED EXPERIMENTS"
echo "=========================================="

# Step 1: Test the fixes
echo "Step 1: Testing bug fixes..."
python test_fixes.py
if [ $? -ne 0 ]; then
    echo "❌ Bug fix tests failed!"
    exit 1
fi

echo "✅ Bug fix tests passed!"
echo ""

# Step 2: Run minimal optimized experiment
echo "Step 2: Running minimal optimized experiment..."
echo "Using debug configuration with very small dataset..."

python run_optimized_experiments.py \
    --config debug_config \
    --datasets kaggle_solar_plant \
    --max-samples 200 \
    --skip-ablation \
    --skip-sensitivity \
    --skip-computational \
    --skip-cross-dataset \
    --skip-robustness

if [ $? -eq 0 ]; then
    echo "✅ Minimal experiment completed successfully!"
    echo ""
    echo "Results should be available in: results/optimized/"
    echo ""
    echo "Next steps:"
    echo "1. Check results in results/optimized/"
    echo "2. Run with larger datasets if needed"
    echo "3. Submit optimized job to cluster"
else
    echo "❌ Minimal experiment failed!"
    echo "Check the logs for detailed error information"
    exit 1
fi

echo "=========================================="
echo "LOCAL TESTING COMPLETED SUCCESSFULLY!"
echo "=========================================="