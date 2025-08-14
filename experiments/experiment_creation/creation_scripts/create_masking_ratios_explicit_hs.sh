#!/bin/bash
# Base folder for the experiment
BASE="/home/ubuntu/smiles_encoder/experiments/masking_ratios_explicit_hs"
# Path to the base config file
CONFIG_TEMPLATE="/home/ubuntu/smiles_encoder/config/pretrain_mlm.yaml"

# Only dataset and model size for this experiment
dataset="fabikru/chembl-2025-randomized-smiles-cleaned-explicit-hs"
size="15M"

# Define the masking ratios to test
masking_ratios=("0.1" "0.15" "0.2" "0.3" "0.4" "0.5" "0.6")

# Extract dataset key (everything after the last slash)
dataset_key="${dataset##*/}"

# Set learning rate and batch size for explicit-hs dataset
eta=0.005776
batch_size=256  # Rounded from 279.721463

# Maximum per-device batch size for the 15M model
max_bs=512

# Determine per-device batch size and gradient accumulation steps.
if [ "$batch_size" -le "$max_bs" ]; then
    per_device_bs=$batch_size
    grad_steps=1
else
    per_device_bs=$max_bs
    grad_steps=$(( (batch_size + max_bs - 1) / max_bs ))
fi

# Compute the effective batch size and new evaluation/save steps.
effective_bs=$(( per_device_bs * grad_steps ))
new_steps=$(( 500000 / effective_bs ))

# Loop over each masking ratio
for masking in "${masking_ratios[@]}"; do
    # Create a final directory structure: BASE/dataset_key/size/masking_<masking>
    final_dir="$BASE/$dataset_key/$size/masking_${masking}"
    mkdir -p "$final_dir"

    # Create a temporary file for config modifications.
    temp_file=$(mktemp)

    # Escape any '/' and '&' characters in dataset for sed.
    dataset_escaped=$(printf '%s\n' "$dataset" | sed 's/[\/&]/\\&/g')

    # Replace parameters in the config template with the desired values.
    sed -e "s|^[[:space:]]*masking_probability:.*|masking_probability: $masking|" \
        -e "s|^[[:space:]]*dataset_name:.*|dataset_name: $dataset_escaped|" \
        -e "s|^[[:space:]]*result_folder_path:.*|result_folder_path: \"$final_dir\"|" \
        -e "s|^[[:space:]]*learning_rate:.*|    learning_rate: $eta|" \
        -e "s|^[[:space:]]*per_device_train_batch_size:.*|    per_device_train_batch_size: $per_device_bs|" \
        -e "s|^[[:space:]]*per_device_eval_batch_size:.*|    per_device_eval_batch_size: $per_device_bs|" \
        -e "s|^[[:space:]]*gradient_accumulation_steps:.*|    gradient_accumulation_steps: $grad_steps|" \
        -e "s|^[[:space:]]*eval_steps:.*|    eval_steps: $new_steps|" \
        -e "s|^[[:space:]]*save_steps:.*|    save_steps: $new_steps|" \
        "$CONFIG_TEMPLATE" > "$temp_file"

    # Find the line number where modern_bert_config starts.
    mod_line=$(grep -n "^[[:space:]]*modern_bert_config:" "$temp_file" | cut -d: -f1 | head -n1)
    if [ -z "$mod_line" ]; then
        echo "modern_bert_config: not found in config file. Skipping $final_dir."
        rm "$temp_file"
        continue
    fi

    # Extract everything before the modern_bert_config block.
    head -n $((mod_line - 1)) "$temp_file" > "${temp_file}.head"

    # Define the new modern_bert_config block for the 15M model.
    new_mod_config="modern_bert_config:
    hidden_size: 384
    num_attention_heads: 6
    intermediate_size: 576
    num_hidden_layers: 12
    global_attn_every_n_layers: 1
    max_position_embeddings: 502"

    # Write the final configuration file: header + a blank line + the new modern_bert_config block.
    {
        cat "${temp_file}.head"
        echo ""
        echo "$new_mod_config"
    } > "$final_dir/config.yaml"

    # Clean up temporary files.
    rm "$temp_file" "${temp_file}.head"
done

echo "Folder tree and adapted config files created successfully." 