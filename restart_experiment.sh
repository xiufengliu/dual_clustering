#!/bin/bash
# Restart the comprehensive evaluation with all fixes applied

echo "=== COMPREHENSIVE FIXES APPLIED ==="
echo "✓ Fixed dataset configuration - now using all 12 available datasets"
echo "✓ Fixed ARIMA/SARIMA .values attribute compatibility"
echo "✓ Fixed deep learning models prediction length mismatch"
echo "✓ Fixed boolean indexing error in neutrosophic transformer"
echo "✓ Fixed KMeans tolerance parameter type conversion"
echo "✓ Added robust error handling and data validation"
echo "✓ Fixed sequence creation for LSTM, CNN-LSTM, N-BEATS"
echo ""

echo "=== Killing current job if running ==="
# Get current job ID
CURRENT_JOB=$(bjobs -w | grep dual_cluster_comprehensive | awk '{print $1}')
if [ ! -z "$CURRENT_JOB" ]; then
    echo "Killing job $CURRENT_JOB"
    bkill $CURRENT_JOB
    sleep 5
fi

echo "=== Cleaning up old output files ==="
rm -f gpu_comprehensive_*.out gpu_comprehensive_*.err

echo "=== Submitting new job with comprehensive fixes ==="
bsub < submit_comprehensive_eval.sh

echo "=== Monitoring new job ==="
sleep 5
bjobs -w

echo ""
echo "=== Available Datasets ==="
echo "The experiment will now run on these datasets:"
echo "- gefcom2014_energy, gefcom2014_solar, gefcom2014_wind"
echo "- kaggle_solar_plant, kaggle_wind_power"
echo "- nrel_canada_wind, nrel_solar, nrel_wind"
echo "- uk_sheffield_solar, entso_e_load"
echo "- entso_e_solar, entso_e_wind (legacy)"
echo ""
echo "Monitor progress with:"
echo "  bjobs"
echo "  tail -f gpu_comprehensive_*.out"
echo "  ./monitor_job.sh"
