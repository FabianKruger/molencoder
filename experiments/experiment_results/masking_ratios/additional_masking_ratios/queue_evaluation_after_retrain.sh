#!/bin/bash

# Queue script to run masking ratio evaluation after 70% model retraining finishes
# Current retrain queue PID
RETRAIN_QUEUE_PID=2704911

echo "Waiting for 70% model retraining queue (PID: $RETRAIN_QUEUE_PID) to finish..."
echo "Started monitoring at $(date)"

# Wait for the retrain queue process to finish
while kill -0 "$RETRAIN_QUEUE_PID" 2>/dev/null; do
    echo "$(date): Retrain queue still running (PID: $RETRAIN_QUEUE_PID)..."
    sleep 300  # Check every 5 minutes
done

echo "$(date): 70% model retraining completed! Starting masking ratio evaluation..."

# Change to the correct directory and run the evaluation
cd /home/ubuntu/smiles_encoder/experiments/additional_masking_ratios

echo "$(date): Running masking ratio evaluation for all models (0.1 to 0.9)..."
./run_evaluation.sh

echo "$(date): Masking ratio evaluation queue completed!"