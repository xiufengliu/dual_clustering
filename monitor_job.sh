#!/bin/bash
# Monitor the optimized experiment job

JOB_ID=25616006
OUTPUT_FILE="gpu_comprehensive_${JOB_ID}.out"
ERROR_FILE="gpu_comprehensive_${JOB_ID}.err"

echo "=== Job Monitoring Script ==="
echo "Job ID: $JOB_ID"
echo "Output file: $OUTPUT_FILE"
echo "Error file: $ERROR_FILE"
echo

# Check job status
echo "=== Current Job Status ==="
bjobs -l $JOB_ID | grep -E "Status|PENDING TIME|MEMORY USAGE|CPU USAGE"
echo

# Check if output file exists and show last few lines
if [ -f "$OUTPUT_FILE" ]; then
    echo "=== Latest Output (last 20 lines) ==="
    tail -n 20 "$OUTPUT_FILE"
    echo

    # Check for progress indicators
    echo "=== Progress Indicators ==="
    grep -E "Running|Completed|Evaluating|Dataset|experiment" "$OUTPUT_FILE" | tail -n 10
    echo
else
    echo "Output file $OUTPUT_FILE does not exist yet."
fi

# Check if error file exists and show any errors
if [ -f "$ERROR_FILE" ]; then
    if [ -s "$ERROR_FILE" ]; then
        echo "=== Error Messages ==="
        cat "$ERROR_FILE"
        echo
    else
        echo "No errors reported in $ERROR_FILE"
    fi
else
    echo "Error file $ERROR_FILE does not exist yet."
fi

# Provide instructions for checking results
echo
echo "=== Instructions ==="
echo "1. To check job status again: bjobs -l $JOB_ID"
echo "2. To view full output: cat $OUTPUT_FILE"
echo "3. To view errors: cat $ERROR_FILE"
echo "4. To cancel job if needed: bkill $JOB_ID"
echo "5. To check results after completion: ls -la results/comprehensive/"
echo
echo "Expected completion time: 30-60 minutes from job start"
echo "Run this script again to update status: ./monitor_job.sh"

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
