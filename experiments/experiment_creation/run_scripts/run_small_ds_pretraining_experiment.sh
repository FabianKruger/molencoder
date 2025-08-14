#!/bin/bash
# Source conda to enable conda commands and activate the appropriate environment.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/ubuntu/miniconda3/envs/encoder

# Base folder where the config files are located.
BASE="/home/ubuntu/smiles_encoder/experiments/small_ds_pretraining"

# Find each config.yaml in the directory tree.
configs=$(find "$BASE" -type f -name "config.yaml")

# Loop over the found configuration files and launch the training job for each one sequentially.
for config in $configs; do
    echo "Launching job for config: $config"
    accelerate launch --config_file /home/ubuntu/.cache/huggingface/accelerate/default_config.yaml /home/ubuntu/smiles_encoder/scripts/models/pretrain_mlm.py --config "$config"
    echo "Job for $config finished."
done

echo "All jobs have been launched and completed." 