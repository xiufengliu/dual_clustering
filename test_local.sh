#!/bin/bash
# Simple local test script using local Python

# Source bashrc to set up local Python environment
source ~/.bashrc

echo "=== Testing local setup ==="
echo "Python version:"
python --version
echo "Python path:"
which python

echo "=== Installing dependencies ==="
pip install --user -r requirements.txt

echo "=== Testing imports ==="
python -c "
import sys
sys.path.append('.')
try:
    from src.framework.forecasting_framework import NeutrosophicForecastingFramework
    from src.models.baseline_models import BaselineForecasters
    from src.evaluation.metrics import ForecastingMetrics
    print('✓ All imports successful')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

echo "=== Local test completed ==="
echo "If successful, you can submit to the cluster with:"
echo "bsub < submit_comprehensive_eval.sh"
