#!/bin/bash

# Extended Advanced Model Comparison Study Runner Script
# This script runs the extended comparison with additional datasets

echo "$(date): Starting extended comparison study..."

# Activate the polaris conda environment
echo "Activating polaris conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate polaris

# Set the working directory
cd /home/ubuntu/smiles_encoder/experiments/result_analysis/extended_advanced_comparison_to_other_models

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/extended_comparison_study_${TIMESTAMP}.log"

echo "Starting extended model comparison study at $(date)"
echo "Logs will be saved to: $LOG_FILE"
echo "You can monitor progress with: tail -f $LOG_FILE"

# Run the experiment with nohup
nohup python run_comparison_study.py > "$LOG_FILE" 2>&1 &

# Get the process ID
PID=$!
echo "Process started with PID: $PID"
echo "To kill the process if needed: kill $PID"

# Save PID to file for easy reference
echo $PID > logs/extended_comparison_study.pid

echo "Extended comparison study started successfully!"
echo "Monitor with: tail -f $LOG_FILE" 