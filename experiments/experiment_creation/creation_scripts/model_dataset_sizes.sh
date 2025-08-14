#!/bin/bash
# Base folder for the experiment
BASE="/home/ubuntu/smiles_encoder/model_and_dataset_sizes"
# Path to the base config file
CONFIG_TEMPLATE="/home/ubuntu/smiles_encoder/config/pretrain_mlm.yaml"

# Use a single masking probability of 0.3 for this experiment
masking=0.3

# List of datasets (folder name will be derived from the part after the last slash)
datasets=(
    "fabikru/half-of-chembl-2025-randomized-smiles-cleaned"
    "fabikru/chembl-2025-randomized-smiles-cleaned"
    "fabikru/pubchem_and_chembl-2025-randomized-smiles-cleaned"
)

# Model sizes (each corresponds to a specific model parameter count)
# "5M"    -> 4,547,666
# "15M"   -> 15,229,522
# "150M"  -> 110,985,298
sizes=("5M" "15M" "150M")

# Loop over each dataset
for dataset in "${datasets[@]}"; do
    # Derive folder name by stripping everything before (and including) the last slash
    dataset_key="${dataset##*/}"
    dataset_dir="$BASE/$dataset_key"
    mkdir -p "$dataset_dir"

    # Loop over each model size
    for size in "${sizes[@]}"; do
        final_dir="$dataset_dir/$size"
        mkdir -p "$final_dir"

        # Determine optimal learning rate (eta) and optimal batch size based on the combination.
        # Original calculated optimal batch sizes:
        #   half-of-chembl:      293.845364 -> 256 (nearest power of 2)
        #   chembl:              436.436081 -> 512
        #   pubchem_and_chembl:  4230.575150 -> 4096
        if [ "$size" = "5M" ]; then
            if [ "$dataset_key" = "half-of-chembl-2025-randomized-smiles-cleaned" ]; then
                eta=0.008141
                batch_size=256
            elif [ "$dataset_key" = "chembl-2025-randomized-smiles-cleaned" ]; then
                eta=0.010073
                batch_size=512
            elif [ "$dataset_key" = "pubchem_and_chembl-2025-randomized-smiles-cleaned" ]; then
                eta=0.032227
                batch_size=4096
            fi
        elif [ "$size" = "15M" ]; then
            if [ "$dataset_key" = "half-of-chembl-2025-randomized-smiles-cleaned" ]; then
                eta=0.003439
                batch_size=256
            elif [ "$dataset_key" = "chembl-2025-randomized-smiles-cleaned" ]; then
                eta=0.004255
                batch_size=512
            elif [ "$dataset_key" = "pubchem_and_chembl-2025-randomized-smiles-cleaned" ]; then
                eta=0.013613
                batch_size=4096
            fi
        elif [ "$size" = "150M" ]; then
            if [ "$dataset_key" = "half-of-chembl-2025-randomized-smiles-cleaned" ]; then
                eta=0.000834
                batch_size=256
            elif [ "$dataset_key" = "chembl-2025-randomized-smiles-cleaned" ]; then
                eta=0.001033
                batch_size=512
            elif [ "$dataset_key" = "pubchem_and_chembl-2025-randomized-smiles-cleaned" ]; then
                eta=0.003303
                batch_size=4096
            fi
        fi

        # Determine maximum per-device batch size allowed in memory based on the model size.
        # For the 5M model: max_bs = 1028
        # For the 15M model: max_bs = 512
        # For the 150M model: max_bs = 256
        if [ "$size" = "5M" ]; then
            max_bs=1028
        elif [ "$size" = "15M" ]; then
            max_bs=512
        elif [ "$size" = "150M" ]; then
            max_bs=256
        fi

        # Determine per-device batch size and gradient accumulation steps.
        # Use the maximum allowed per device if the optimal batch size exceeds max_bs.
        if [ "$batch_size" -le "$max_bs" ]; then
            per_device_bs=$batch_size
            grad_steps=1
        else
            per_device_bs=$max_bs
            # Ceiling division: (batch_size + max_bs - 1) / max_bs
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
        # Using [[:space:]]* to match optional leading whitespace.
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

        # Define the new modern_bert_config block based on the model size.
        if [ "$size" = "5M" ]; then
            new_mod_config="modern_bert_config:
    hidden_size: 256
    num_attention_heads: 4
    intermediate_size: 384
    num_hidden_layers: 8
    global_attn_every_n_layers: 1
    max_position_embeddings: 502"
        elif [ "$size" = "15M" ]; then
            new_mod_config="modern_bert_config:
    hidden_size: 384
    num_attention_heads: 6
    intermediate_size: 576
    num_hidden_layers: 12
    global_attn_every_n_layers: 1
    max_position_embeddings: 502"
        elif [ "$size" = "150M" ]; then
            new_mod_config="modern_bert_config:
    hidden_size: 768
    num_attention_heads: 12
    intermediate_size: 1152
    num_hidden_layers: 22
    global_attn_every_n_layers: 1
    max_position_embeddings: 502"
        fi

        # Write out the final configuration file: header + a blank line + the new modern_bert_config block.
        {
            cat "${temp_file}.head"
            echo ""
            echo "$new_mod_config"
        } > "$final_dir/config.yaml"

        # Clean up temporary files.
        rm "$temp_file" "${temp_file}.head"
    done
done

echo "Folder tree and adapted config files created successfully."
