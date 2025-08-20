# MolEncoder Fine-tuning Script

Fine-tune the [MolEncoder](https://huggingface.co/fabikru/MolEncoder) model on molecular property prediction tasks.

## Overview

`train.py` is a comprehensive script that fine-tunes MolEncoder for both regression and classification tasks. It automatically handles data preprocessing, hyperparameter optimization through cross-validation, and supports multi-task learning with robust NaN handling.

## Quick Start

```bash
python train.py --data_train data.csv --config config.cfg
```

## Input Requirements

### Data CSV Format
**Required columns:**
- `smiles`: SMILES strings representing molecular structures
- `Results0`, `Results1`, etc.: Target properties (at least one required)

**Optional columns:**
- `desc0`, `desc1`, etc.: Descriptors (ignored during training)

### Configuration File
Simple `.cfg` file specifying the task type and output location:
```ini
[DEFAULT]
classification = false  # true for classification, false for regression
output_dir = ./my_model  # directory where model will be saved
```

### Missing Values (NaN)
The script intelligently handles missing values by **excluding them from loss computation** while keeping all samples. No data is wasted through imputation or removal.

**Example:** A sample with `Results0=0.5, Results1=NaN` will contribute to learning `Results0` but not `Results1`.

## What It Does

1. **Cross-validation**: Finds optimal training epochs (1-500) using 5-fold CV
2. **Preprocessing**: 
   - Tokenizes SMILES with MolEncoder tokenizer
   - Applies robust scaling for regression (median + IQR)
   - Maps classification labels to 0-based indices
3. **Training**: Fine-tunes the full MolEncoder model
4. **NaN handling**: Masks missing labels from gradient computation
5. **Saves**: Model, tokenizer, scaler, and training logs

## Supported Tasks

| **Task Type** | **Description** | **Example** |
|---------------|-----------------|-------------|
| **Single Regression** | One continuous property | Solubility prediction |
| **Multi Regression** | Multiple continuous properties | ADME properties |
| **Single Classification** | One categorical property | Active/Inactive |
| **Multi-task Classification** | Multiple independent tasks | Multiple assays |

## Output

Creates a model directory with:
- **Trained model**: `model.safetensors`, `config.json`
- **Tokenizer**: MolEncoder tokenizer files  
- **Preprocessing**: `label_scaler.pkl` (regression), `label_columns.txt`
- **Logs**: `training.log` with detailed training information

## Example Data Files

This directory includes working examples:
- `example_data_regression.csv` - Multi-target regression
- `example_data_classification.csv` - Single classification  
- `example_data_multitask_classification.csv` - Multi-task classification
- `example_data_with_nans.csv` - Regression with missing values
- `example_multitask_classification_with_nans.csv` - Classification with missing values

## Requirements

```bash
pip install torch transformers datasets scikit-learn pandas numpy
```

**Note**: This script is specifically designed for MolEncoder and uses the `fabikru/MolEncoder` checkpoint.
