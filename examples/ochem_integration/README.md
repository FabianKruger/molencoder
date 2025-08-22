# MolEncoder Training and Prediction Pipeline

Fine-tune and apply the [MolEncoder](https://huggingface.co/fabikru/MolEncoder) model for molecular property prediction. File structures are designed for integration into the OCHEM platform ([OCHEM](https://ochem.eu/home/show.do)).

- **`train.py`**: Fine-tune MolEncoder model on custom data
- **`apply.py`**: Make predictions with trained models

## Quick Start

```bash
# Train a model
python train.py --config config.cfg

# Make predictions
python apply.py --config config.cfg
```

## Data Format

### Training Data (`train.py`)
**CSV file with:**
- `smiles`: SMILES strings
- `Results0`, `Results1`, etc.: Target properties/labels
- Optional: `desc0`, `desc1`, etc.: Descriptors (ignored during training)

### Prediction Data (`apply.py`)
**CSV file with:**
- `smiles`: SMILES strings only (no target labels needed)

## Configuration

### For Training (`train.py`)
```ini
[DEFAULT]
classification = false  # true for classification, false for regression
weighted_loss = false  # true for class imbalance weighted loss (classification only)
train_data_file = example_data/training_data.csv # path to training data csv file
model_tar_path = ./my_model.tar.gz # location to save the trained model file
```

### For Prediction (`apply.py`)
```ini
[DEFAULT]
model_tar_path = ./my_model.tar.gz  # path to trained model file
apply_data_file = example_data/new_molecules.csv
result_file = predictions.csv
```

**Note:** Both training and prediction parameters can be saved in the same config file. Each script will only use the parameters it needs and ignore the others.

## Examples

```bash
# Regression
python train.py --config example_data/example_config_regression.cfg
python apply.py --config example_data/example_config_regression.cfg

# Classification
python train.py --config example_data/example_config_binary_classification.cfg
python apply.py --config example_data/example_config_binary_classification.cfg

# Multi-task
python train.py --config example_data/example_config_multitask_classification.cfg
python apply.py --config example_data/example_config_multitask_classification.cfg
```

## Requirements

```bash
pip install -r requirements.txt
```
