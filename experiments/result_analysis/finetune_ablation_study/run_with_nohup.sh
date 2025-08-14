#!/bin/bash

# Hyperparameter Ablation Study Runner with nohup
# This script runs the experiment in the background with proper logging

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$SCRIPT_DIR/logs/ablation_study_$TIMESTAMP.log"
ERROR_FILE="$SCRIPT_DIR/logs/ablation_study_$TIMESTAMP.err"

echo "Starting hyperparameter ablation study at $(date)"
echo "Logs will be written to: $LOG_FILE"
echo "Errors will be written to: $ERROR_FILE"
echo "Process will run in background with nohup"

# Change to the script directory
cd "$SCRIPT_DIR"

# Activate conda environment and run the experiment with nohup
nohup bash -c "
source ~/miniconda3/etc/profile.d/conda.sh
conda activate encoder
cd '$SCRIPT_DIR'
python hyperparameter_ablation_study.py --config config.yaml
" > "$LOG_FILE" 2> "$ERROR_FILE" &

# Get the process ID
PID=$!
echo "Process started with PID: $PID"
echo "$PID" > "$SCRIPT_DIR/logs/ablation_study.pid"

echo "To monitor progress, run:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check for errors, run:"
echo "  tail -f $ERROR_FILE"
echo ""
echo "To check if process is still running:"
echo "  ps -p $PID"
echo ""
echo "To kill the process if needed:"
echo "  kill $PID"

# Wait a few seconds and check if process started successfully
sleep 5
if ps -p $PID > /dev/null; then
    echo "✅ Process is running successfully!"
else
    echo "❌ Process failed to start. Check error log:"
    cat "$ERROR_FILE"
    exit 1
fi 