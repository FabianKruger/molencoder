#!/bin/bash

# Queue script to run retrain_70_percent.sh after MoleculeACE comparison finishes
# Current MoleculeACE comparison PID
MOLECULEACE_PID=2691539

echo "Waiting for MoleculeACE comparison (PID: $MOLECULEACE_PID) to finish..."
echo "Started monitoring at $(date)"

# Wait for the MoleculeACE comparison to finish
while kill -0 "$MOLECULEACE_PID" 2>/dev/null; do
    echo "$(date): MoleculeACE comparison still running (PID: $MOLECULEACE_PID)..."
    sleep 300  # Check every 5 minutes
done

echo "$(date): MoleculeACE comparison finished! Starting 70% model retraining..."

# Change to the correct directory and run the retrain script
cd /home/ubuntu/smiles_encoder/experiments/additional_masking_ratios

echo "$(date): Running retrain_70_percent.sh..."
./retrain_70_percent.sh

echo "$(date): 70% model retraining queue completed!"