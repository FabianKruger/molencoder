# MolEncoder: Towards Optimal Masked Language Modeling for Molecules

This repository contains the code and experiments for the paper **"MolEncoder: Towards Optimal Masked Language Modeling for Molecules"**.

MolEncoder is an encoder-only transformer specifically designed for SMILES representation learning using masked language modeling. The model provides state-of-the-art performance for molecular property prediction tasks and is available on the Hugging Face model hub as [`fabikru/MolEncoder`](https://huggingface.co/fabikru/MolEncoder).

## Quick Start

MolEncoder is available on Hugging Face and can be used directly with the `transformers` library. 

For complete usage examples and tutorials:

- **Fill-Mask for Molecular Completion**: See `examples/fill_mask.ipynb`
- **Fine-tuning for Regression Tasks**: See `examples/regression_finetune.ipynb`
- **Fine-tuning for Classification Tasks**: See `examples/classification_finetune.ipynb`

These notebooks contain complete tutorials with all necessary code, dependencies, and explanations.

## Repository Structure

### Core Implementation (`src/`)

The `src/` directory contains all the core code used in the paper:

- **`molencoder/`** - Main package containing:
  - **`tokenizers/`** - Tokenizers for SMILES
  - **`finetune/`** - Fine-tuning utilities for downstream tasks
  - **`evaluation/`** - Cross-validation, metrics, and baseline implementations
  - **`scripts/`** - Data preprocessing and model training scripts
  - **`utils/`** - Helper functions and callbacks

### Examples (`examples/`)

**Start here!** - The primary way to use MolEncoder. Contains practical tutorials:

- **`fill_mask.ipynb`** - Demonstrates masked token prediction for molecular completion
- **`regression_finetune.ipynb`** - Complete tutorial for fine-tuning MolEncoder on your own molecular regression tasks
- **`classification_finetune.ipynb`**: - Complete tutorial for fine-tuning MolEncoder on your own molecular classification tasks
  
These notebooks contain all the necessary code and dependencies. Simply open them and follow along!

### Experiments (`experiments/`)

Contains all experimental data and analysis from the paper:

- **`experiment_creation/`** - Shell scripts to create and run experiments
- **`experiment_results/`** - Raw experimental results and data
- **`result_analysis/`** - Jupyter notebooks analyzing experimental results

## Citation

If you use MolEncoder in your research, please cite:

```bibtex
@article{Krüger_Österbacka_Kabeshov_Engkvist_Tetko_2025, title={MolEncoder: Towards Optimal Masked Language Modeling for Molecules}, DOI={10.26434/chemrxiv-2025-h4w9d}, journal={ChemRxiv}, author={Krüger, Fabian Per and Österbacka, Nicklas and Kabeshov, Mikhail and Engkvist, Ola and Tetko, Igor}, year={2025}}  This content is a preprint and has not been peer-reviewed.
```

## Links

- **Model on Hugging Face**: [`fabikru/MolEncoder`](https://huggingface.co/fabikru/MolEncoder)
- **Paper**: [ChemRxiv](https://doi.org/10.26434/chemrxiv-2025-h4w9d)

## License

This project is licensed under the terms specified in the `LICENSE` file.
