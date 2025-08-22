#!/usr/bin/env python3
"""
Apply trained MolEncoder model to make predictions on new data.

This script loads a trained MolEncoder model and applies it to make predictions
on new SMILES data. It handles both regression and classification tasks,
including multi-task scenarios.

Usage:
    python apply.py --config config.cfg

The config.cfg file should specify:
- 'apply_data_file': Path to CSV file with SMILES to predict
- 'result_file': Path where prediction results will be saved
- 'model_tar_path': Path to the trained model tar file
- Other training parameters to determine task type and setup
"""

import argparse
import configparser
import json
import logging
import pickle
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModel,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from safetensors.torch import load_file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiTaskClassificationModel(nn.Module):
    """Multi-task classification model with shared encoder and multiple heads."""
    
    def __init__(self, model_name: str, num_labels_per_task: List[int]):
        super(MultiTaskClassificationModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.classifiers = nn.ModuleList([
            nn.Linear(self.base_model.config.hidden_size, num_labels)
            for num_labels in num_labels_per_task
        ])
        self.num_tasks = len(num_labels_per_task)
        self.num_labels_per_task = num_labels_per_task
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Always use CLS token
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        
        return {'logits': logits}


class LabelScaler:
    """Scale labels using robust scaling (median and interquartile range)."""
    
    def __init__(self, labels: np.ndarray):
        """Initialize scaler with training labels."""
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        
        # Compute statistics ignoring NaN values
        self.medians = np.nanmedian(labels, axis=0)
        q1 = np.nanpercentile(labels, 25, axis=0)
        q3 = np.nanpercentile(labels, 75, axis=0)
        self.iqrs = q3 - q1
        
        # Handle zero IQR (all values are the same)
        zero_iqr_mask = self.iqrs == 0
        if np.any(zero_iqr_mask):
            logger.warning(f"Found {np.sum(zero_iqr_mask)} label columns with zero IQR. Using std instead.")
            stds = np.nanstd(labels, axis=0)
            self.iqrs[zero_iqr_mask] = stds[zero_iqr_mask]
            # If std is also zero, use 1 to avoid division by zero
            self.iqrs[self.iqrs == 0] = 1.0
    
    def scale_labels(self, labels: np.ndarray) -> np.ndarray:
        """Scale the labels using median and IQR, preserving NaN values."""
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        # Scaling preserves NaN values automatically in numpy
        scaled_labels = (labels - self.medians) / self.iqrs
        return scaled_labels.squeeze() if scaled_labels.shape[1] == 1 else scaled_labels
    
    def inverse_scale(self, scaled_labels: np.ndarray) -> np.ndarray:
        """Rescale predictions back to original scale, preserving NaN values."""
        if scaled_labels.ndim == 1:
            scaled_labels = scaled_labels.reshape(-1, 1)
        # Inverse scaling preserves NaN values automatically in numpy
        original_scale = scaled_labels * self.iqrs + self.medians
        return original_scale.squeeze() if original_scale.shape[1] == 1 else original_scale


def load_model_and_metadata(model_dir: Path) -> Tuple[Any, Dict[str, Any], AutoTokenizer, Optional[LabelScaler], List[str]]:
    """Load trained model, metadata, tokenizer, and scaler."""
    logger.info(f"Loading model from {model_dir}")
    
    # Load metadata
    metadata_path = model_dir / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Model metadata: {metadata}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load label columns
    label_columns_path = model_dir / 'label_columns.txt'
    if not label_columns_path.exists():
        raise FileNotFoundError(f"Label columns file not found: {label_columns_path}")
    
    with open(label_columns_path, 'r') as f:
        label_columns = [line.strip() for line in f.readlines()]
    
    # Load label scaler if regression
    label_scaler = None
    if not metadata['is_classification']:
        scaler_path = model_dir / 'label_scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                label_scaler = pickle.load(f)
            logger.info("Loaded label scaler for regression")
        else:
            logger.warning("No label scaler found for regression task")
    
    # Load model based on task type
    if metadata['is_classification'] and metadata.get('is_multi_task', False):
        # Multi-task classification - we need to manually load since it's a custom model
        logger.info("Loading multi-task classification model")
        model = MultiTaskClassificationModel(
            metadata['model_name'], 
            metadata['num_labels_info']
        )
        # Load the saved state dict - Hugging Face Trainer saves as pytorch_model.bin or model.safetensors
        pytorch_model_path = model_dir / 'pytorch_model.bin'
        safetensors_model_path = model_dir / 'model.safetensors'
        
        if safetensors_model_path.exists():
            # Load safetensors format
            state_dict = load_file(safetensors_model_path)
            model.load_state_dict(state_dict)
        elif pytorch_model_path.exists():
            # Load pytorch .bin format
            state_dict = torch.load(pytorch_model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Model weights not found in {model_dir}")
    else:
        # Single-task classification or regression - use standard Hugging Face loading
        logger.info(f"Loading {'classification' if metadata['is_classification'] else 'regression'} model")
        if metadata['is_classification']:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir, 
                problem_type="regression"
            )
    
    return model, metadata, tokenizer, label_scaler, label_columns


def tokenize_data(data: pd.DataFrame, tokenizer: AutoTokenizer) -> Dataset:
    """Tokenize SMILES data for prediction."""
    logger.info(f"Tokenizing {len(data)} SMILES...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['smiles'],
            truncation=True,
            padding=False,  # Let DataCollator handle padding
            max_length=502
        )
    
    dataset = Dataset.from_pandas(data[['smiles']])
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset


def make_predictions(
    model: Any, 
    dataset: Dataset, 
    tokenizer: AutoTokenizer,
    metadata: Dict[str, Any]
) -> np.ndarray:
    """Make predictions using the trained model."""
    logger.info("Making predictions...")
    
    # Remove the 'smiles' column from dataset as it's not needed for prediction
    prediction_dataset = dataset.remove_columns(['smiles'])
    
    # Create a temporary trainer for prediction (much cleaner approach)
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # Use temporary directory that gets automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        # Minimal training args just for prediction
        training_args = TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=32,
            dataloader_drop_last=False,
            fp16=torch.cuda.is_available(),
            bf16=False,  # Disable bf16 for compatibility - enable if GPU supports bf16
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
        )
        
        # Use trainer.predict() - much simpler and more robust
        predictions = trainer.predict(prediction_dataset)
        logits = predictions.predictions
        
        if metadata['is_classification']:
            if metadata.get('is_multi_task', False):
                # Multi-task classification: get class predictions and probabilities for each task
                if isinstance(logits, list):
                    # logits is a list of arrays for each task
                    task_predictions = []
                    task_probabilities = []
                    for task_logits in logits:
                        # Get class predictions
                        task_preds = np.argmax(task_logits, axis=1)
                        task_predictions.append(task_preds)
                        # Get probabilities using softmax
                        task_probs = torch.softmax(torch.tensor(task_logits), dim=1).numpy()
                        task_probabilities.append(task_probs)
                    # Stack predictions: (num_tasks, batch_size) -> (batch_size, num_tasks)
                    final_predictions = np.array(task_predictions).T
                    # Stack probabilities: list of (batch_size, num_classes) -> (batch_size, num_tasks, num_classes)
                    final_probabilities = task_probabilities
                else:
                    # Handle case where logits is a single array but multi-task
                    final_predictions = np.argmax(logits, axis=-1)
                    if final_predictions.ndim == 1:
                        final_predictions = final_predictions.reshape(-1, 1)
                    # Compute probabilities
                    final_probabilities = [torch.softmax(torch.tensor(logits), dim=-1).numpy()]
            else:
                # Single-task classification: get class predictions and probabilities
                final_predictions = np.argmax(logits, axis=1)
                if final_predictions.ndim == 1:
                    final_predictions = final_predictions.reshape(-1, 1)
                # Compute probabilities using softmax
                probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
                final_probabilities = [probabilities]
        else:
            # Regression: use logits directly
            final_predictions = logits
            if final_predictions.ndim == 1:
                final_predictions = final_predictions.reshape(-1, 1)
            final_probabilities = None
        
        logger.info(f"Generated predictions with shape: {final_predictions.shape}")
        return final_predictions, final_probabilities


def save_predictions(
    predictions: np.ndarray,
    label_columns: List[str],
    result_file: str,
    label_scaler: Optional[LabelScaler] = None,
    is_classification: bool = False,
    probabilities: Optional[List[np.ndarray]] = None
):
    """Save predictions to CSV file."""
    logger.info(f"Saving predictions to {result_file}")
    
    if is_classification and probabilities is not None:
        # For classification: save probabilities directly in Result columns
        logger.info("Saving classification probabilities in Result columns")
        
        # Determine number of tasks
        num_tasks = len(probabilities)
        
        # Create DataFrame with probabilities
        df_data = {}
        
        for task_idx, task_probs in enumerate(probabilities):
            num_classes = task_probs.shape[1]
            
            if num_classes == 2:
                # Binary classification: save probability of positive class (class 1)
                df_data[f"Result{task_idx}"] = task_probs[:, 1]
            else:
                # Multi-class: save all class probabilities as a formatted string
                prob_lists = []
                for sample_idx in range(len(task_probs)):
                    prob_list = task_probs[sample_idx].tolist()
                    # Format as string list for CSV (you can change this format if needed)
                    prob_str = str(prob_list)
                    prob_lists.append(prob_str)
                df_data[f"Result{task_idx}"] = prob_lists
        
        df = pd.DataFrame(df_data)
        
    else:
        # For regression: apply inverse scaling and save continuous values
        if not is_classification and label_scaler is not None:
            logger.info("Applying inverse scaling to regression predictions")
            predictions = label_scaler.inverse_scale(predictions)
        
        # Create DataFrame with proper column names
        num_targets = predictions.shape[1] if predictions.ndim > 1 else 1
        
        if num_targets == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Create column names: Result0, Result1, etc.
        column_names = [f"Result{i}" for i in range(num_targets)]
        df = pd.DataFrame(predictions, columns=column_names)
    
    # Save to CSV
    df.to_csv(result_file, index=False)
    logger.info(f"Saved {len(df)} predictions to {result_file}")


def main():
    parser = argparse.ArgumentParser(description='Apply trained MolEncoder model for predictions')
    parser.add_argument('--config', required=True, help='Path to configuration .cfg file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = configparser.ConfigParser()
    config.read(args.config)
    
    # Get required parameters from config
    try:
        apply_data_file = config.get('DEFAULT', 'apply_data_file')
        result_file = config.get('DEFAULT', 'result_file')
        model_tar_path = config.get('DEFAULT', 'model_tar_path')
    except (configparser.NoOptionError, KeyError) as e:
        raise ValueError(f"Missing required parameter in config file: {e}")
    
    # Check if model tar file exists
    if not Path(model_tar_path).exists():
        raise FileNotFoundError(f"Model tar file not found: {model_tar_path}")
    
    # Extract model tar file to temporary directory
    
    temp_model_dir = tempfile.mkdtemp(prefix='molencoder_model_')
    output_dir = Path(temp_model_dir)
    
    logger.info(f"Extracting model from {model_tar_path} to temporary directory")
    with tarfile.open(model_tar_path, 'r:gz') as tar:
        tar.extractall(path=output_dir)
    
    try:
        # Load data to predict
        logger.info(f"Loading data from {apply_data_file}")
        data = pd.read_csv(apply_data_file)
        
        if 'smiles' not in data.columns:
            raise ValueError("Input data must contain a 'smiles' column")
        
        logger.info(f"Loaded {len(data)} samples for prediction")
        
        # Load model and metadata
        model, metadata, tokenizer, label_scaler, label_columns = load_model_and_metadata(output_dir)
        
        # Tokenize data
        dataset = tokenize_data(data, tokenizer)
        
        # Make predictions (device handling is automatic in trainer.predict())
        predictions, probabilities = make_predictions(model, dataset, tokenizer, metadata)
        
        # Save predictions
        save_predictions(
            predictions, 
            label_columns, 
            result_file,
            label_scaler,
            metadata['is_classification'],
            probabilities
        )
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise
    finally:
        # Clean up temporary directory
        if 'temp_model_dir' in locals():
            logger.info("Cleaning up temporary model directory")
            shutil.rmtree(temp_model_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
