#!/bin/bash
# Restart the comprehensive evaluation with fixes

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

echo "=== Submitting new job with fixes ==="
bsub < submit_comprehensive_eval.sh

echo "=== Monitoring new job ==="
sleep 5
bjobs -w

echo ""
echo "Monitor progress with:"
echo "  bjobs"
echo "  tail -f gpu_comprehensive_*.out"
echo "  ./monitor_job.sh"
