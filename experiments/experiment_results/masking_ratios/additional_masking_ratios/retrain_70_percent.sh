#!/bin/bash

# Script to retrain only the 70% masking model
# Source conda to enable conda commands and activate the appropriate environment.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate polaris

# Specific config for 70% model
CONFIG="/home/ubuntu/smiles_encoder/experiments/additional_masking_ratios/half-of-chembl-2025-randomized-smiles-cleaned/15M/masking_0.7/config.yaml"

echo "Retraining 70% masking model..."
echo "Config: $CONFIG"
echo "=========================================="

# Launch the training job
accelerate launch --config_file /home/ubuntu/.cache/huggingface/accelerate/default_config.yaml /home/ubuntu/smiles_encoder/scripts/models/pretrain_mlm.py --config "$CONFIG"

echo "=========================================="
echo "70% model retraining completed!"