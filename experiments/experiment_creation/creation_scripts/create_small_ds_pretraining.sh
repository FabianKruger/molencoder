#!/bin/bash
# Base folder for the experiment
BASE="/home/ubuntu/smiles_encoder/experiments/small_ds_pretraining"
# Path to the base config file
CONFIG_TEMPLATE="/home/ubuntu/smiles_encoder/config/pretrain_mlm.yaml"

# Use a single masking probability of 0.3 for this experiment
masking=0.3

# List of datasets (folder name will be derived from the part after the last slash)
datasets=(
    "fabikru/quarter-of-chembl-2025-randomized-smiles-cleaned"
    "fabikru/eighth-of-chembl-2025-randomized-smiles-cleaned"
)

# Model size (15M model)
size="15M"

# Loop over each dataset
for dataset in "${datasets[@]}"; do
    # Derive folder name by stripping everything before (and including) the last slash
    dataset_key="${dataset##*/}"
    dataset_dir="$BASE/$dataset_key"
    mkdir -p "$dataset_dir"

    final_dir="$dataset_dir/$size"
    mkdir -p "$final_dir"

    # Determine optimal learning rate (eta) and optimal batch size based on the dataset.
    if [ "$dataset_key" = "quarter-of-chembl-2025-randomized-smiles-cleaned" ]; then
        eta=0.002780
        batch_size=256
    elif [ "$dataset_key" = "eighth-of-chembl-2025-randomized-smiles-cleaned" ]; then
        eta=0.002247
        batch_size=128
    fi

    # Determine maximum per-device batch size allowed in memory based on the model size.
    max_bs=512

    # Determine per-device batch size and gradient accumulation steps.
    if [ "$batch_size" -le "$max_bs" ]; then
        per_device_bs=$batch_size
        grad_steps=1
    else
        per_device_bs=$max_bs
        grad_steps=$(( (batch_size + max_bs - 1) / max_bs ))
    fi

    # Compute the effective batch size and the new eval/save steps.
    effective_bs=$(( per_device_bs * grad_steps ))
    new_steps=$(( 500000 / effective_bs ))

    # Create a temporary file for the config modifications.
    temp_file=$(mktemp)

    # Escape any '/' and '&' characters in the dataset variable for sed.
    dataset_escaped=$(printf '%s\n' "$dataset" | sed 's/[\/&]/\\&/g')

    # Replace parameters in the config template.
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

    # Write out the final configuration file: header + a blank line + the new modern_bert_config block.
    {
        cat "${temp_file}.head"
        echo ""
        echo "$new_mod_config"
    } > "$final_dir/config.yaml"

    # Clean up temporary files.
    rm "$temp_file" "${temp_file}.head"
done

echo "Folder tree and adapted config files created successfully for small datasets." 