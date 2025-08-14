#!/bin/bash
# Source conda to enable conda commands
source ~/miniconda3/etc/profile.d/conda.sh
conda activate polaris

# Base folder where the config files are located
BASE="/home/ubuntu/smiles_encoder/model_and_dataset_sizes"

# Find each config.yaml in the directory tree
configs=$(find "$BASE" -type f -name "config.yaml")

for config in $configs; do
    echo "Launching job for config: $config"
    accelerate launch --config_file /home/ubuntu/.cache/huggingface/accelerate/default_config.yaml /home/ubuntu/smiles_encoder/scripts/models/pretrain_mlm.py --config "$config"
    echo "Job for $config finished."
done

echo "All jobs have been launched and completed."
