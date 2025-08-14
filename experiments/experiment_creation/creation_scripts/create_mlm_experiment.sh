#!/bin/bash
# Base folder where the experiment will be built
BASE="/home/ubuntu/smiles_encoder/mlm_experiment"
# Path to the base config file
CONFIG_TEMPLATE="/home/ubuntu/smiles_encoder/scripts/models/config.yaml"

# First-level folder names and corresponding masking_probability values.
mlm_names=("mlm015" "mlm03" "mlm045")
mlm_probs=(0.15 0.3 0.45)

# Second-level dataset names (full names for the config; folder name is derived by removing prefix).
datasets=(
    "fabikru/half-of-chembl-2025-randomized-smiles-cleaned"
    "fabikru/chembl-2025-randomized-smiles-cleaned"
    "fabikru/pubchem_and_chembl-2025-randomized-smiles-cleaned"
)

# Third-level folder names.
sizes=("5M" "15M" "150M")

# Loop over first-level directories.
for i in "${!mlm_names[@]}"; do
    mlm="${mlm_names[$i]}"
    masking="${mlm_probs[$i]}"
    mlm_dir="$BASE/$mlm"
    mkdir -p "$mlm_dir"

    # Loop over second-level dataset directories.
    for dataset in "${datasets[@]}"; do
        # Derive folder name by stripping everything before (and including) the last slash.
        folder_name="${dataset##*/}"
        dataset_dir="$mlm_dir/$folder_name"
        mkdir -p "$dataset_dir"

        # Loop over third-level directories.
        for size in "${sizes[@]}"; do
            final_dir="$dataset_dir/$size"
            mkdir -p "$final_dir"

            # Create a temporary file to work on the config.
            temp_file=$(mktemp)

            # Escape any '/' and '&' in the dataset variable for sed.
            dataset_escaped=$(printf '%s\n' "$dataset" | sed 's/[\/&]/\\&/g')

            # Replace masking_probability, dataset_name, and result_folder_path lines.
            sed -e "s|^masking_probability:.*|masking_probability: $masking|" \
                -e "s|^dataset_name:.*|dataset_name: $dataset_escaped|" \
                -e "s|^result_folder_path:.*|result_folder_path: \"$final_dir\"|" \
                "$CONFIG_TEMPLATE" > "$temp_file"

            # Find the line number where modern_bert_config starts.
            mod_line=$(grep -n "^modern_bert_config:" "$temp_file" | cut -d: -f1 | head -n1)
            if [ -z "$mod_line" ]; then
                echo "modern_bert_config: not found in config file. Skipping $final_dir."
                rm "$temp_file"
                continue
            fi

            # Extract everything before the modern_bert_config block.
            head -n $((mod_line - 1)) "$temp_file" > "${temp_file}.head"

            # Determine the new modern_bert_config block based on the size.
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

            # Write out the final config: the head part, a blank line, then the new modern_bert_config block.
            {
                cat "${temp_file}.head"
                echo ""
                echo "$new_mod_config"
            } > "$final_dir/config.yaml"

            # Clean up temporary files.
            rm "$temp_file" "${temp_file}.head"
        done
    done
done

echo "Folder tree and adapted config files created successfully."
