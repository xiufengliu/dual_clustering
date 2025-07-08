#!/bin/bash
# Monitor the comprehensive evaluation job

echo "=== Job Status ==="
bjobs -w

echo ""
echo "=== Latest Output (last 20 lines) ==="
tail -20 gpu_comprehensive_*.out

echo ""
echo "=== Any Errors ==="
if [ -f gpu_comprehensive_*.err ]; then
    tail -10 gpu_comprehensive_*.err
else
    echo "No error file found"
fi

echo ""
echo "=== Job Progress Summary ==="
if grep -q "Installing collected packages" gpu_comprehensive_*.out; then
    echo "✓ Dependencies installation in progress..."
fi

if grep -q "Running comprehensive evaluation" gpu_comprehensive_*.out; then
    echo "✓ Comprehensive evaluation started"
fi

if grep -q "Experiment completed" gpu_comprehensive_*.out; then
    echo "✓ Experiment completed successfully!"
fi

echo ""
echo "To monitor in real-time, run:"
echo "watch -n 30 './monitor_job.sh'"
