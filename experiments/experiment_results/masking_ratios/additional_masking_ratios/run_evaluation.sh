#!/bin/bash

LOG_DIR="/home/ubuntu/smiles_encoder/experiments/additional_masking_ratios/evaluation_results"
OUT_LOG="$LOG_DIR/polaris_eval.out"
ERR_LOG="$LOG_DIR/polaris_eval.err"

# Create the results directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "$(date): Starting evaluation of all masking ratio models (0.1 to 0.9)..."

# Set up environment
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Activate the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate polaris

echo "$(date): Running polaris evaluation..."

# Run the Python script with output redirection
python /home/ubuntu/smiles_encoder/src/smilesencoder/scripts/polaris_evaluation.py \
    --config /home/ubuntu/smiles_encoder/experiments/additional_masking_ratios/evaluation_config.yaml \
    > "$OUT_LOG" 2> "$ERR_LOG"

echo "$(date): Evaluation completed. Check logs at:"
echo "  Output log: $OUT_LOG"
echo "  Error log:  $ERR_LOG"
echo "  Results:    $LOG_DIR"