#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate polaris

# Base directory for the hparam insensitivity experiment configs
EXPERIMENT_DIR="/home/ubuntu/smiles_encoder/hparam_insensitivity_experiment"

# Loop over each subfolder in the experiment directory and execute the corresponding config file
for run_dir in "$EXPERIMENT_DIR"/*; do
    if [ -d "$run_dir" ]; then
        echo "Launching job for config in $run_dir"
        accelerate launch --config_file /home/ubuntu/.cache/huggingface/accelerate/default_config.yaml \
            /home/ubuntu/smiles_encoder/scripts/models/pretrain_mlm.py --config "$run_dir/config.yaml"
    fi
done

echo "All jobs have been launched and completed."
