#!/bin/bash

# Monitor the comprehensive evaluation job
JOB_ID=25619656

echo "=== Monitoring Comprehensive Evaluation Job $JOB_ID ==="
echo "Started at: $(date)"
echo

# Function to check job status
check_job_status() {
    bjobs $JOB_ID 2>/dev/null | grep -v JOBID
}

# Function to show recent output
show_recent_output() {
    echo "=== Recent Output ==="
    if [ -f "gpu_comprehensive_${JOB_ID}.out" ]; then
        tail -20 "gpu_comprehensive_${JOB_ID}.out"
    else
        echo "Output file not found yet"
    fi
    echo
}

# Function to show recent errors
show_recent_errors() {
    echo "=== Recent Errors ==="
    if [ -f "gpu_comprehensive_${JOB_ID}.err" ]; then
        tail -10 "gpu_comprehensive_${JOB_ID}.err"
    else
        echo "Error file not found yet"
    fi
    echo
}

# Monitor loop
while true; do
    clear
    echo "=== Job Status Check at $(date) ==="
    
    # Check if job is still running
    STATUS=$(check_job_status)
    if [ -z "$STATUS" ]; then
        echo "Job $JOB_ID has completed or is not found"
        break
    else
        echo "Job Status: $STATUS"
    fi
    
    echo
    show_recent_output
    show_recent_errors
    
    echo "=== Waiting 30 seconds before next check... ==="
    echo "Press Ctrl+C to stop monitoring"
    sleep 30
done

echo "=== Final Status ==="
echo "Job completed at: $(date)"

# Show final results
if [ -f "gpu_comprehensive_${JOB_ID}.out" ]; then
    echo
    echo "=== Final Output (last 50 lines) ==="
    tail -50 "gpu_comprehensive_${JOB_ID}.out"
fi

if [ -f "gpu_comprehensive_${JOB_ID}.err" ]; then
    echo
    echo "=== Final Errors ==="
    cat "gpu_comprehensive_${JOB_ID}.err"
fi

# Check for results
echo
echo "=== Checking Results Directory ==="
if [ -d "results/comprehensive" ]; then
    echo "Results directory contents:"
    ls -la results/comprehensive/
else
    echo "Results directory not found"
fi

echo
echo "=== Monitoring Complete ==="
