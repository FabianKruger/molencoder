# MolEncoder Training and Prediction Pipeline

Complete pipeline for fine-tuning and applying the [MolEncoder](https://huggingface.co/fabikru/MolEncoder) model on molecular property prediction tasks.

## Overview

This directory provides two main scripts:
- **`train.py`**: Fine-tunes MolEncoder for regression and classification tasks
- **`apply.py`**: Applies trained models to make predictions on new data

Both scripts are controlled through simple configuration files and handle all preprocessing automatically.

## Quick Start

```bash
# Train a model
python train.py --config config.cfg

# Make predictions
python apply.py --config config.cfg
```

## Input Requirements

### Data CSV Format
**Required columns:**
- `smiles`: SMILES strings representing molecular structures
- `Results0`, `Results1`, etc.: Target properties (at least one required)

**Optional columns:**
- `desc0`, `desc1`, etc.: Descriptors (ignored during training)

### Configuration File
Simple `.cfg` file specifying all parameters:
```ini
[DEFAULT]
classification = false  # true for classification, false for regression
train_data_file = training_data.csv  # path to training data CSV file
output_dir = ./my_model  # directory where model will be saved

# For predictions (apply.py)
apply_data_file = new_molecules.csv  # path to data for prediction
result_file = predictions.csv  # where to save predictions
```

### Missing Values (NaN)
The script intelligently handles missing values by **excluding them from loss computation** while keeping all samples. No data is wasted through imputation or removal.

**Example:** A sample with `Results0=0.5, Results1=NaN` will contribute to learning `Results0` but not `Results1`.

## Training Pipeline (`train.py`)

1. **Cross-validation**: Finds optimal training epochs (1-500) using 5-fold CV
2. **Preprocessing**: 
   - Tokenizes SMILES with MolEncoder tokenizer
   - Applies robust scaling for regression (median + IQR)
   - Maps classification labels to 0-based indices
3. **Training**: Fine-tunes the full MolEncoder model
4. **NaN handling**: Masks missing labels from gradient computation
5. **Saves**: Model, tokenizer, scaler, and training logs

## Prediction Pipeline (`apply.py`)

1. **Model Loading**: Automatically loads trained model and preprocessing components
2. **Data Processing**: Tokenizes new SMILES using the same tokenizer as training
3. **Inference**: Generates predictions using the trained model
4. **Post-processing**: 
   - **Regression**: Applies inverse scaling to return original units
   - **Classification**: Returns most probable class for each task
5. **Output**: Saves predictions as CSV with `Result0`, `Result1`, etc. columns

## Supported Tasks

| **Task Type** | **Description** | **Example** |
|---------------|-----------------|-------------|
| **Single Regression** | One continuous property | Solubility prediction |
| **Multi Regression** | Multiple continuous properties | ADME properties |
| **Single Classification** | One categorical property | Active/Inactive |
| **Multi-task Classification** | Multiple independent tasks | Multiple assays |

## Output Files

### Training Output (`train.py`)
Creates a model directory with:
- **Trained model**: `model.safetensors`, `config.json`
- **Tokenizer**: MolEncoder tokenizer files  
- **Preprocessing**: `label_scaler.pkl` (regression), `label_columns.txt`
- **Logs**: `training.log` with detailed training information

### Prediction Output (`apply.py`)
Creates a CSV file with predictions:
- **Regression**: Continuous values in original units (after inverse scaling)
- **Classification**: Integer class predictions (0, 1, 2, etc.)
- **Multi-task**: Multiple `Result0`, `Result1`, etc. columns

**Example outputs:**
```csv
# Regression predictions
Result0,Result1
0.279,1.851
0.286,1.844

# Classification predictions  
Result0
2
1
0
```

## Complete Workflow Example

1. **Train a regression model:**
```bash
python train.py --config example_config_regression.cfg
```

2. **Make predictions on new data:**
```bash
python apply.py --config example_config_regression.cfg
```

3. **Check predictions:**
```bash
cat regression_predictions.csv
```

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

## Features

- ✅ **Complete Pipeline**: Training → Prediction in two simple commands
- ✅ **Automatic Scaling**: Handles preprocessing and inverse scaling seamlessly  
- ✅ **Multi-task Support**: Single config handles multiple targets
- ✅ **NaN Robust**: Intelligent handling of missing values
- ✅ **Device Agnostic**: CPU, CUDA, Apple Silicon (MPS) support
- ✅ **Reproducible**: All settings captured in config files

**Note**: These scripts are specifically designed for MolEncoder and use the `fabikru/MolEncoder` checkpoint.
