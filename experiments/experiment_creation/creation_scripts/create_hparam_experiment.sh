#!/bin/bash
# Base directory where the experiment folder will be created.
BASE="/home/ubuntu/smiles_encoder"
EXPERIMENT_DIR="$BASE/hparam_insensitivity_experiment"
mkdir -p "$EXPERIMENT_DIR"

# Source configuration file to copy.
SRC_CONFIG="$BASE/model_and_dataset_sizes/chembl-2025-randomized-smiles-cleaned/5M/config.yaml"

# Define subfolder names, per-device batch sizes, and learning rates.
subfolders=("run1" "run2" "run3" "run4")
# Batch sizes: for run1/run2 -> 256, for run3/run4 -> 1028.
batch_sizes=(256 256 1028 1028)
# Learning rates: for run1/run3 -> 0.1, for run2/run4 -> 0.001.
lrs=(0.1 0.001 0.1 0.001)

# Loop over each subfolder.
for i in "${!subfolders[@]}"; do
    folder="${subfolders[$i]}"
    target_dir="$EXPERIMENT_DIR/$folder"
    mkdir -p "$target_dir"
    
    # Copy the source config file into the subfolder.
    cp "$SRC_CONFIG" "$target_dir/config.yaml"
    
    # Determine the absolute path for the target folder.
    abs_target_dir=$(cd "$target_dir" && pwd)
    
    new_bs="${batch_sizes[$i]}"
    new_lr="${lrs[$i]}"
    
    # Update the configuration file:
    # - Set result_folder_path to the subfolder's absolute path.
    # - Set per_device_train_batch_size and per_device_eval_batch_size.
    # - Set learning_rate.
    sed -i \
        -e "s|^[[:space:]]*result_folder_path:.*|result_folder_path: \"$abs_target_dir\"|" \
        -e "s|^[[:space:]]*per_device_train_batch_size:.*|    per_device_train_batch_size: $new_bs|" \
        -e "s|^[[:space:]]*per_device_eval_batch_size:.*|    per_device_eval_batch_size: $new_bs|" \
        -e "s|^[[:space:]]*learning_rate:.*|    learning_rate: $new_lr|" \
        "$target_dir/config.yaml"
done

echo "hparam_insensitivity_experiment created with updated config files."
