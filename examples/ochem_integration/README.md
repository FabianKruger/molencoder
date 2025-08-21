# MolEncoder Training and Prediction Pipeline

Complete pipeline for fine-tuning and applying the [MolEncoder](https://huggingface.co/fabikru/MolEncoder) model on molecular property prediction tasks using the command line interface.

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
weighted_loss = false  # true to use class-weighted loss (classification only)
train_data_file = example_data/training_data.csv  # path to training data CSV file
output_dir = ./my_model  # directory where model will be saved

# For predictions (apply.py)
apply_data_file = example_data/new_molecules.csv  # path to data for prediction
result_file = predictions.csv  # where to save predictions
```

### Missing Values (NaN)
The script intelligently handles missing values by **excluding them from loss computation** while keeping all samples. No data is wasted through imputation or removal.

**Example:** A sample with `Results0=0.5, Results1=NaN` will contribute to learning `Results0` but not `Results1`.

### Class-Weighted Loss (Classification Only)
For imbalanced classification datasets, you can enable class-weighted loss to automatically balance the learning across different classes:

```ini
[DEFAULT]
classification = true
weighted_loss = true  # Enables class-weighted loss
```

**When to use:**
- **Imbalanced datasets**: When some classes have significantly fewer examples
- **Multi-task classification**: When different tasks have different class distributions
- **Improved recall**: When you need better detection of minority classes

**How it works:**
- Automatically calculates class weights inversely proportional to class frequencies
- Applied per task in multi-task scenarios (each task gets its own weights)
- Uses scikit-learn's `class_weight='balanced'` strategy
- Only affects training; predictions remain unchanged

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
- **Classification**: Probabilities directly in `Result` columns (no separate class predictions)
- **Multi-task**: Multiple `Result0`, `Result1`, etc. columns

**Example outputs:**
```csv
# Regression predictions
Result0,Result1
0.279,1.851
0.286,1.844

# Binary classification predictions (probabilities of positive class)
Result0,Result1
0.74,0.82
0.32,0.15
0.68,0.91

# Multi-class classification predictions (all class probabilities as lists)
Result0,Result1
"[0.32, 0.28, 0.39]","[0.15, 0.65, 0.20]"
"[0.45, 0.42, 0.13]","[0.25, 0.55, 0.20]"
"[0.51, 0.35, 0.14]","[0.30, 0.45, 0.25]"
```

## Complete Workflow Examples

### Basic Regression
1. **Train a regression model:**
```bash
python train.py --config example_data/example_config_regression.cfg
```

2. **Make predictions on new data:**
```bash
python apply.py --config example_data/example_config_regression.cfg
```

3. **Check predictions:**
```bash
cat regression_predictions.csv
```

### Multi-task Classification
1. **Train a multi-task classification model:**
```bash
python train.py --config example_data/example_config_multitask_classification.cfg
```

2. **Make predictions:**
```bash
python apply.py --config example_data/example_config_multitask_classification.cfg
```

### Imbalanced Classification with Weighted Loss
1. **Train with class weighting for imbalanced data:**
```bash
python train.py --config example_data/example_config_weighted_classification.cfg
```

2. **Make predictions:**
```bash
python apply.py --config example_data/example_config_weighted_classification.cfg
```

3. **Compare results with unweighted training:**
```bash
python train.py --config example_data/example_config_binary_classification.cfg
# Compare weighted_classification_predictions.csv vs binary_classification_predictions.csv
```

### Handling Missing Values
1. **Train with datasets containing NaN values:**
```bash
python train.py --config example_data/example_config_regression_with_nans.cfg
python train.py --config example_data/example_config_multitask_classification_with_nans.cfg
```

## Example Data and Config Files

This directory includes working examples in the `example_data/` folder:

**Data Files:**
- `example_data/example_data_regression.csv` - Multi-target regression (2 continuous targets)
- `example_data/example_data_classification.csv` - Multi-class classification (classes 0, 1, 2)
- `example_data/example_data_binary_classification.csv` - Binary classification (classes 0, 1)
- `example_data/example_data_multitask_classification.csv` - Multi-task classification (2 classification tasks)
- `example_data/example_data_with_nans.csv` - Regression with missing values
- `example_data/example_multitask_classification_with_nans.csv` - Multi-task classification with missing values

**Config Files:**
| **Config File** | **Task Type** | **Data File** | **Use Case** |
|----------------|---------------|---------------|--------------|
| `example_data/example_config_regression.cfg` | Regression | `example_data_regression.csv` | Multi-target regression |
| `example_data/example_config_classification.cfg` | Multi-class | `example_data_classification.csv` | Single multi-class task |
| `example_data/example_config_binary_classification.cfg` | Binary | `example_data_binary_classification.csv` | Simple binary classification |
| `example_data/example_config_multitask_classification.cfg` | Multi-task | `example_data_multitask_classification.csv` | Multiple classification tasks |
| `example_data/example_config_weighted_classification.cfg` | Binary (weighted) | `example_data_binary_classification.csv` | Imbalanced data handling |
| `example_data/example_config_regression_with_nans.cfg` | Regression | `example_data_with_nans.csv` | Regression with missing values |
| `example_data/example_config_multitask_classification_with_nans.cfg` | Multi-task | `example_multitask_classification_with_nans.csv` | Multi-task with missing values |

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch transformers datasets scikit-learn pandas numpy accelerate schedulefree
```

## Features

- ✅ **Complete Pipeline**: Training → Prediction in two simple commands
- ✅ **Automatic Scaling**: Handles preprocessing and inverse scaling seamlessly  
- ✅ **Multi-task Support**: Single config handles multiple targets
- ✅ **NaN Robust**: Intelligent handling of missing values
- ✅ **Class-Weighted Loss**: Automatic handling of imbalanced classification datasets
- ✅ **Device Agnostic**: CPU, CUDA, Apple Silicon (MPS) support
- ✅ **Reproducible**: All settings captured in config files

**Note**: These scripts are specifically designed for MolEncoder and use the `fabikru/MolEncoder` checkpoint.
