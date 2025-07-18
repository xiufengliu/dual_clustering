#!/bin/bash

# Validated Job Submission Script
# Comprehensive validation before submitting dual clustering experiments

set -e  # Exit on any error

echo "=== Validated Job Submission System ==="
echo "Started at: $(date)"
echo

# Configuration
CONFIG_NAME="benchmark_config"
VALIDATION_DIR="results/validation"
LOG_FILE="$VALIDATION_DIR/job_submission_$(date +%Y%m%d_%H%M%S).log"

# Create validation directory
mkdir -p "$VALIDATION_DIR"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check if validation passed
check_validation_status() {
    local report_file="$1"
    if [ -f "$report_file" ]; then
        local status=$(python -c "import json; print(json.load(open('$report_file'))['overall_status'])")
        if [ "$status" = "True" ]; then
            return 0
        else
            return 1
        fi
    else
        return 1
    fi
}

log_message "Starting validated job submission process"

# Step 1: Run pre-flight validation
log_message "Step 1: Running pre-flight validation"
if python scripts/pre_flight_checks.py --config "$CONFIG_NAME" --verbose >> "$LOG_FILE" 2>&1; then
    log_message "✅ Pre-flight validation passed"
else
    log_message "❌ Pre-flight validation failed"
    echo "Pre-flight validation failed. Check log: $LOG_FILE"
    exit 1
fi

# Step 2: Validate individual datasets
log_message "Step 2: Validating individual datasets"

# List of critical datasets to validate
DATASETS=(
    "entso_e_solar"
    "nrel_solar" 
    "kaggle_solar_plant"
    "kaggle_wind_power"
    "gefcom2014_energy"
    "uk_sheffield_solar"
)

VALIDATION_PASSED=0
VALIDATION_TOTAL=0

for dataset in "${DATASETS[@]}"; do
    log_message "Validating dataset: $dataset"
    VALIDATION_TOTAL=$((VALIDATION_TOTAL + 1))
    
    if python scripts/validate_single_dataset.py --dataset "$dataset" --config "$CONFIG_NAME" --validate-all >> "$LOG_FILE" 2>&1; then
        log_message "✅ Dataset $dataset validation passed"
        VALIDATION_PASSED=$((VALIDATION_PASSED + 1))
    else
        log_message "❌ Dataset $dataset validation failed"
    fi
done

log_message "Dataset validation results: $VALIDATION_PASSED/$VALIDATION_TOTAL passed"

# Require at least 80% of datasets to pass
MIN_REQUIRED=$((VALIDATION_TOTAL * 80 / 100))
if [ "$VALIDATION_PASSED" -lt "$MIN_REQUIRED" ]; then
    log_message "❌ Insufficient datasets passed validation ($VALIDATION_PASSED < $MIN_REQUIRED required)"
    echo "Dataset validation failed. Check log: $LOG_FILE"
    exit 1
fi

# Step 3: Check system resources
log_message "Step 3: Checking system resources"

# Check available memory
AVAILABLE_MEMORY=$(free -g | awk '/^Mem:/{print $7}')
if [ "$AVAILABLE_MEMORY" -lt 8 ]; then
    log_message "⚠️  Low available memory: ${AVAILABLE_MEMORY}GB (recommended: 8GB+)"
fi

# Check GPU availability (if on login node)
if command -v nvidia-smi &> /dev/null; then
    GPU_STATUS=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    log_message "GPU utilization: ${GPU_STATUS}%"
fi

# Step 4: Backup previous results
log_message "Step 4: Backing up previous results"
if [ -d "results/comprehensive" ]; then
    BACKUP_DIR="results/backup/comprehensive_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    cp -r results/comprehensive/* "$BACKUP_DIR/" 2>/dev/null || true
    log_message "Previous results backed up to: $BACKUP_DIR"
fi

# Step 5: Submit job with validation metadata
log_message "Step 5: Submitting validated job"

# Create validation summary
VALIDATION_SUMMARY="$VALIDATION_DIR/submission_validation_summary.json"
cat > "$VALIDATION_SUMMARY" << EOF
{
    "submission_time": "$(date -Iseconds)",
    "config_name": "$CONFIG_NAME",
    "pre_flight_passed": true,
    "datasets_validated": $VALIDATION_PASSED,
    "datasets_total": $VALIDATION_TOTAL,
    "validation_rate": $(echo "scale=2; $VALIDATION_PASSED * 100 / $VALIDATION_TOTAL" | bc),
    "system_memory_gb": $(free -g | awk '/^Mem:/{print $2}'),
    "available_memory_gb": $AVAILABLE_MEMORY,
    "validation_log": "$LOG_FILE"
}
EOF

# Submit the job
log_message "Submitting comprehensive evaluation job..."
JOB_OUTPUT=$(bsub < submit_comprehensive_eval.sh 2>&1)
JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Job <\K[0-9]+')

if [ -n "$JOB_ID" ]; then
    log_message "✅ Job submitted successfully: ID $JOB_ID"
    
    # Update validation summary with job ID
    python -c "
import json
with open('$VALIDATION_SUMMARY', 'r') as f:
    data = json.load(f)
data['job_id'] = '$JOB_ID'
with open('$VALIDATION_SUMMARY', 'w') as f:
    json.dump(data, f, indent=2)
"
    
    # Create monitoring script for this specific job
    cat > "monitor_job_$JOB_ID.sh" << EOF
#!/bin/bash
# Auto-generated monitoring script for job $JOB_ID

echo "=== Monitoring Job $JOB_ID ==="
echo "Submitted at: $(date)"
echo "Validation summary: $VALIDATION_SUMMARY"
echo

while true; do
    clear
    echo "=== Job Status at \$(date) ==="
    
    # Check job status
    STATUS=\$(bjobs $JOB_ID 2>/dev/null | grep -v JOBID)
    if [ -z "\$STATUS" ]; then
        echo "Job $JOB_ID has completed or is not found"
        break
    else
        echo "Job Status: \$STATUS"
    fi
    
    echo
    echo "=== Recent Output ==="
    if [ -f "gpu_comprehensive_$JOB_ID.out" ]; then
        tail -20 "gpu_comprehensive_$JOB_ID.out"
    else
        echo "Output file not found yet"
    fi
    
    echo
    echo "=== Recent Errors ==="
    if [ -f "gpu_comprehensive_$JOB_ID.err" ]; then
        tail -10 "gpu_comprehensive_$JOB_ID.err"
    else
        echo "Error file not found yet"
    fi
    
    echo
    echo "Press Ctrl+C to stop monitoring"
    sleep 30
done

echo "=== Job Completed ==="
echo "Final status check at: \$(date)"

# Show final results
if [ -f "gpu_comprehensive_$JOB_ID.out" ]; then
    echo
    echo "=== Final Output (last 50 lines) ==="
    tail -50 "gpu_comprehensive_$JOB_ID.out"
fi

if [ -f "gpu_comprehensive_$JOB_ID.err" ]; then
    echo
    echo "=== Final Errors ==="
    cat "gpu_comprehensive_$JOB_ID.err"
fi
EOF
    
    chmod +x "monitor_job_$JOB_ID.sh"
    
    echo
    echo "=== Job Submission Summary ==="
    echo "Job ID: $JOB_ID"
    echo "Validation Summary: $VALIDATION_SUMMARY"
    echo "Monitoring Script: monitor_job_$JOB_ID.sh"
    echo "Log File: $LOG_FILE"
    echo
    echo "To monitor the job:"
    echo "  bjobs $JOB_ID"
    echo "  ./monitor_job_$JOB_ID.sh"
    echo
    echo "To check output:"
    echo "  tail -f gpu_comprehensive_$JOB_ID.out"
    echo "  tail -f gpu_comprehensive_$JOB_ID.err"
    
else
    log_message "❌ Job submission failed"
    echo "Job submission failed. Output:"
    echo "$JOB_OUTPUT"
    exit 1
fi

log_message "Validated job submission completed successfully"
echo
echo "✅ Validated job submission completed successfully"
echo "Job ID: $JOB_ID"
