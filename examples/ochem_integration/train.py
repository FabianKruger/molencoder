#!/usr/bin/env python3
"""
MolEncoder Fine-tuning Script for OChem Integration

This script fine-tunes the MolEncoder model on molecular data for both regression 
and classification tasks, supporting single-task and multi-task scenarios.

Usage:
    python train.py --data_train data_train.csv --config config.cfg

The data_train.csv file should contain:
- 'smiles' column: SMILES strings
- 'Result0', 'Result1', etc.: target labels (can be multiple for multi-task)
- Optional 'desc0', 'desc1', etc.: descriptor columns (ignored)

The config.cfg file should specify:
- 'classification': True for classification, False for regression
- 'model_tar_path': Path where the trained model tar file will be saved
- Other hyperparameters can be added as needed
"""

import argparse
import configparser
import json
import logging
import pickle
import random
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, classification_report
)
from sklearn.model_selection import KFold
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# Set up basic logging (will be reconfigured in main() with file output)
logger = logging.getLogger(__name__)


def set_reproducible_seed(seed: int = 42):
    """Set reproducible random seeds for all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Use transformers set_seed which also handles additional libraries
    set_seed(seed)
    logger.info(f"Set reproducible seed to {seed}")


class NaNAwareRegressionTrainer(Trainer):
    """Custom trainer that handles NaN values in regression by masking them in loss computation."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation that excludes NaN values from regression loss using MSE with ignore.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Create mask for valid (non-NaN) values
        valid_mask = ~torch.isnan(labels)
        
        if not valid_mask.any():
            # If all values are NaN, return zero loss (no gradient contribution)
            loss = torch.tensor(0.0, device=labels.device, requires_grad=True)
        else:
            # Replace NaN labels with dummy values (won't affect gradients due to masking)
            labels_safe = torch.where(valid_mask, labels, torch.tensor(0.0, device=labels.device))
            
            # Compute element-wise MSE losses (no NaN in computation now)
            loss_fct = nn.MSELoss(reduction='none')
            # Ensure logits and labels have the same shape
            if logits.dim() == 2 and logits.size(1) == 1:
                logits = logits.squeeze(-1)  # Convert (batch_size, 1) to (batch_size,)
            element_losses = loss_fct(logits, labels_safe)
            
            # Zero out losses for positions that had NaN labels (no gradient flow)
            masked_losses = element_losses * valid_mask.float()
            
            # Compute mean only over valid labels
            loss = masked_losses.sum() / valid_mask.sum()
        
        return (loss, outputs) if return_outputs else loss


class NaNAwareClassificationTrainer(Trainer):
    """Custom trainer that handles NaN values in classification by replacing NaN with ignore index."""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        """
        Initialize trainer with optional class weights.
        
        Args:
            class_weights: List of tensors, one per task. Each tensor contains weights for each class.
                          For single-task: [tensor([weight_class0, weight_class1, ...])]
                          For multi-task: [task0_weights, task1_weights, ...]
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation that handles NaN values by setting them to ignore index (-100).
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        # Replace NaN values with -100 (ignore index)
        nan_mask = torch.isnan(labels.float())
        labels_masked = labels.clone()
        labels_masked[nan_mask] = -100
        
        # Put labels back for standard loss computation
        inputs["labels"] = labels_masked
        
        # Use standard loss computation with ignore_index
        logits = outputs.get("logits")
        
        if isinstance(logits, list):
            # Multi-task classification - custom loss computation
            total_loss = 0
            valid_tasks = 0
            
            for task_idx, task_logits in enumerate(logits):
                # Get labels for this task
                if labels_masked.dim() > 1:
                    task_labels = labels_masked[:, task_idx]
                else:
                    task_labels = labels_masked
                
                # Only compute loss if there are valid labels
                valid_indices = task_labels != -100
                if valid_indices.any():
                    # Use class weights if available
                    task_weights = None
                    if self.class_weights and task_idx < len(self.class_weights):
                        task_weights = self.class_weights[task_idx].to(task_logits.device)
                    
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=task_weights)
                    task_loss = loss_fct(task_logits, task_labels.long())
                    total_loss += task_loss
                    valid_tasks += 1
            
            # Average loss across valid tasks
            loss = total_loss / max(valid_tasks, 1) if valid_tasks > 0 else torch.tensor(0.0, device=labels.device, requires_grad=True)
        else:
            # Single-task classification - use standard CrossEntropyLoss with ignore_index
            weights = None
            if self.class_weights and len(self.class_weights) > 0:
                weights = self.class_weights[0].to(logits.device)  # Use first (and only) task weights
                
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=weights)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels_masked.view(-1).long())
        
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels: np.ndarray, is_multi_task: bool = False, beta: float = 0.9999) -> List[torch.Tensor]:
    """
    Compute class weights using Effective Number of Samples approach.
    
    This method is more conservative than inverse frequency weighting and works better
    for extremely imbalanced datasets.
    
    Args:
        labels: numpy array of shape (n_samples, n_tasks) or (n_samples,) for single task
        is_multi_task: whether this is multi-task classification
        beta: smoothing parameter for effective number calculation (0.9999 is typical)
    
    Returns:
        List of torch tensors, one per task, containing class weights
        
    Reference:
        "Class-Balanced Loss Based on Effective Number of Samples" 
        (Cui et al., CVPR 2019)
    """
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    
    n_tasks = labels.shape[1]
    class_weights = []
    
    logger.info(f"Computing class weights using Effective Number of Samples (β={beta}):")
    
    for task_idx in range(n_tasks):
        task_labels = labels[:, task_idx]
        
        # Remove NaN values for weight computation
        valid_mask = ~np.isnan(task_labels)
        if not valid_mask.any():
            logger.warning(f"Task {task_idx}: No valid labels, using equal weights")
            # Default to equal weights if no valid labels
            class_weights.append(torch.tensor([1.0, 1.0]))
            continue
        
        valid_labels = task_labels[valid_mask]
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        
        # Compute effective number of samples for each class
        effective_numbers = {}
        weights = {}
        
        for label, count in zip(unique_labels, counts):
            # Effective number: (1 - β^n) / (1 - β)
            if beta == 0:
                effective_num = count  # No reweighting
            else:
                effective_num = (1.0 - beta**count) / (1.0 - beta)
            
            # Weight is inverse of effective number
            weights[int(label)] = 1.0 / effective_num
            effective_numbers[int(label)] = effective_num
        
        # Normalize weights to prevent extreme values
        weight_values = list(weights.values())
        weight_sum = sum(weight_values)
        n_classes = len(weight_values)
        
        # Normalize so that weights sum to n_classes (maintains relative ratios)
        for label in weights:
            weights[label] = weights[label] * n_classes / weight_sum
        
        # Create weight tensor for all classes (0, 1, 2, ...)
        max_class = int(max(unique_labels))
        weight_tensor = torch.ones(max_class + 1)
        
        for class_idx in range(max_class + 1):
            if class_idx in weights:
                weight_tensor[class_idx] = weights[class_idx]
        
        class_weights.append(weight_tensor)
        
        # Log the weights and effective numbers
        pos_samples = counts[unique_labels == 1][0] if 1 in unique_labels else 0
        neg_samples = counts[unique_labels == 0][0] if 0 in unique_labels else 0
        pos_weight = weights.get(1, 1.0)
        neg_weight = weights.get(0, 1.0)
        pos_eff = effective_numbers.get(1, 0)
        neg_eff = effective_numbers.get(0, 0)
        
        logger.info(f"  Task {task_idx}: {neg_samples} negative, {pos_samples} positive samples")
        logger.info(f"  Task {task_idx}: effective numbers = [neg: {neg_eff:.1f}, pos: {pos_eff:.1f}]")
        logger.info(f"  Task {task_idx}: weights = [neg: {neg_weight:.3f}, pos: {pos_weight:.3f}]")
    
    return class_weights


class MultiTaskClassificationModel(nn.Module):
    """Custom model for multi-task classification with shared encoder and multiple heads."""
    
    def __init__(self, model_name: str, num_labels_per_task: List[int]):
        super(MultiTaskClassificationModel, self).__init__()
        # Load the base model (encoder only, without classification head)
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Create separate classification heads for each task
        self.classifiers = nn.ModuleList([
            nn.Linear(self.base_model.config.hidden_size, num_labels)
            for num_labels in num_labels_per_task
        ])
        
        self.num_tasks = len(num_labels_per_task)
        self.num_labels_per_task = num_labels_per_task
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get outputs from base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use CLS token (first token) from last hidden state
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Get logits from each classification head
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            total_loss = 0
            
            for i, task_logits in enumerate(logits):
                # Labels should be structured as [task0_labels, task1_labels, ...]
                task_labels = labels[:, i] if labels.dim() > 1 else labels
                task_loss = loss_fct(task_logits, task_labels.long())
                total_loss += task_loss
            
            loss = total_loss / self.num_tasks  # Average loss across tasks
        
        # Return in the format expected by Trainer
        return {
            'loss': loss,
            'logits': logits,  # List of logits for each task
        }


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


class BestEpochTracker(TrainerCallback):
    """Callback to track the best epoch based on evaluation loss."""
    
    def __init__(self):
        self.best_eval_loss = float("inf")
        self.best_epoch = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control
        
        current_loss = metrics.get("eval_loss")
        # Only update if current loss is valid (not None and not NaN) and better than previous best
        if (current_loss is not None and 
            not np.isnan(current_loss) and 
            np.isfinite(current_loss) and 
            current_loss < self.best_eval_loss):
            self.best_eval_loss = current_loss
            self.best_epoch = metrics.get("epoch")
        return control


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from .cfg file."""
    
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)
    
    config = {}
    
    # Read from DEFAULT section or first available section
    section_name = 'DEFAULT' if 'DEFAULT' in config_parser else config_parser.sections()[0] if config_parser.sections() else None
    
    if section_name is None:
        raise ValueError("No valid section found in config file")
    
    section = config_parser[section_name] if section_name != 'DEFAULT' else config_parser.defaults()
    
    for key, value in section.items():
        # Try to convert to appropriate type
        if isinstance(value, str):
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.replace('.', '').replace('-', '').isdigit():
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass
        
        config[key] = value
    
    logger.info(f"Loaded config: {config}")
    return config


def load_and_preprocess_data(data_path: str, is_classification: bool) -> Tuple[Dataset, Optional[LabelScaler], List[str], Any]:
    """Load and preprocess the training data."""
    logger.info(f"Loading data from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Check required columns
    if 'smiles' not in df.columns:
        raise ValueError("Data must contain 'smiles' column")
    
    # Find label columns (Result0, Result1, etc.)
    label_columns = [col for col in df.columns if col.startswith('Result')]
    if not label_columns:
        raise ValueError("Data must contain at least one Result column (Result0, Result1, etc.)")
    
    label_columns = sorted(label_columns)  # Ensure consistent ordering
    logger.info(f"Found label columns: {label_columns}")
    
    # Extract labels
    labels = df[label_columns].values
    if labels.shape[1] == 1:
        labels = labels.flatten()
    
    # Handle NaN values by reporting and keeping them for loss masking
    if labels.ndim > 1:
        # Multi-task: report NaN statistics
        nan_mask = np.isnan(labels)
        if nan_mask.any():
            nan_counts = nan_mask.sum(axis=0)
            logger.info(f"Found NaN values in labels:")
            for i, col in enumerate(label_columns):
                if nan_counts[i] > 0:
                    logger.info(f"  {col}: {nan_counts[i]}/{len(labels)} NaN label values")
                    logger.info(f"  These {nan_counts[i]} NaN labels will be excluded from loss computation (samples kept)")
        else:
            logger.info("No NaN values found in labels")
    else:
        # Single-task: report NaN statistics
        nan_mask = np.isnan(labels)
        if nan_mask.any():
            nan_count = nan_mask.sum()
            logger.info(f"Found {nan_count}/{len(labels)} NaN label values")
            logger.info(f"These {nan_count} NaN labels will be excluded from loss computation (samples kept)")
        else:
            logger.info("No NaN values found in labels")
    
    # Keep all data, NaNs will be handled in loss computation
    df_clean = df.reset_index(drop=True)
    labels_clean = labels
    
    logger.info(f"Using all {len(df_clean)} samples (NaN values will be masked in loss)")
    
    # Prepare labels for model
    label_scaler = None
    num_labels_info = 1
    
    if is_classification:
        # For classification, ensure labels are integers
        if labels_clean.ndim > 1:
            # Multi-task classification: handle each task separately
            logger.info(f"Multi-task classification with {labels_clean.shape[1]} tasks")
            
            num_labels_per_task = []
            processed_labels = []
            
            for task_idx in range(labels_clean.shape[1]):
                task_labels = labels_clean[:, task_idx]
                
                # Get unique non-NaN labels
                valid_labels = task_labels[~np.isnan(task_labels)]
                unique_labels = np.unique(valid_labels)
                num_labels_for_task = len(unique_labels)
                num_labels_per_task.append(num_labels_for_task)
                
                logger.info(f"Task {task_idx} ({label_columns[task_idx]}): {num_labels_for_task} classes: {unique_labels}")
                
                # Ensure labels start from 0 for this task, keep NaN as NaN
                label_mapping = {old: new for new, old in enumerate(sorted(unique_labels))}
                mapped_labels = np.full_like(task_labels, np.nan)
                for i, label in enumerate(task_labels):
                    if not np.isnan(label):
                        mapped_labels[i] = label_mapping[label]
                processed_labels.append(mapped_labels)
            
            # Stack labels: shape (n_samples, n_tasks)
            labels_clean = np.column_stack(processed_labels)
            num_labels_info = num_labels_per_task
            
        else:
            # Single-task classification
            unique_labels = np.unique(labels_clean)
            num_labels = len(unique_labels)
            logger.info(f"Single-task classification with {num_labels} classes: {unique_labels}")
            # Ensure labels start from 0
            label_mapping = {old: new for new, old in enumerate(sorted(unique_labels))}
            labels_clean = np.array([label_mapping[label] for label in labels_clean])
            num_labels_info = num_labels
    else:
        # For regression, apply scaling
        label_scaler = LabelScaler(labels_clean)
        labels_clean = label_scaler.scale_labels(labels_clean)
        num_labels_info = labels_clean.shape[1] if labels_clean.ndim > 1 else 1
        logger.info(f"Regression task with {num_labels_info} target(s)")
    
    # Create dataset
    dataset_dict = {
        'smiles': df_clean['smiles'].tolist(),
        'labels': labels_clean.tolist()
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    return dataset, label_scaler, label_columns, num_labels_info


def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """Tokenize the SMILES strings in the dataset."""
    def tokenize_function(examples):
        return tokenizer(examples["smiles"], truncation=True, max_length=502)
    
    return dataset.map(tokenize_function, batched=True)


def find_optimal_epochs(
    dataset: Dataset, 
    model_name: str, 
    tokenizer, 
    num_labels_info: Any,
    is_classification: bool,
    weighted_loss: bool = False,
    n_splits: int = 5, 
    max_epochs: int = 100
) -> int:
    """Find optimal number of epochs using cross-validation."""
    logger.info(f"Finding optimal epochs using {n_splits}-fold cross-validation...")
    
    data_collator = DataCollatorWithPadding(tokenizer)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_epochs = []
    
    indices = np.arange(len(dataset))
    
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(indices)):
        logger.info(f"Training fold {fold_num + 1}/{n_splits}")
        
        # Create fold datasets
        train_fold = dataset.select(train_idx.tolist())
        val_fold = dataset.select(val_idx.tolist())
        
        # Initialize model
        if is_classification:
            if isinstance(num_labels_info, list):
                # Multi-task classification
                model = MultiTaskClassificationModel(model_name, num_labels_info)
            else:
                # Single-task classification
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=num_labels_info
                )
        else:
            # Regression
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels_info, problem_type="regression"
            )
        
        best_epoch_tracker = BestEpochTracker()
        early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            training_args = TrainingArguments(
                output_dir=temp_dir,
                logging_dir=temp_dir,
                num_train_epochs=max_epochs,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
                learning_rate=8e-4,
                weight_decay=1e-5,
                warmup_steps=100,
                optim="schedule_free_adamw",
                lr_scheduler_type="constant",
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-8,
                fp16=torch.cuda.is_available(),  # Use fp16 only on CUDA
                bf16=False,  # Disable bf16 for compatibility across environments - enable if GPU supports bf16
                eval_strategy="epoch",
                save_strategy="no",
                max_grad_norm=1.0,
                load_best_model_at_end=False,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                logging_steps=10,
            )
            
            # Use appropriate NaN-aware trainer
            if is_classification:
                # Compute class weights for this fold if weighted loss is enabled
                fold_class_weights = None
                if weighted_loss:
                    fold_labels = np.array([train_fold[i]['labels'] for i in range(len(train_fold))])
                    fold_class_weights = compute_class_weights(fold_labels, is_multi_task=isinstance(num_labels_info, list))
                
                trainer = NaNAwareClassificationTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_fold,
                    eval_dataset=val_fold,
                    data_collator=data_collator,
                    callbacks=[early_stopping, best_epoch_tracker],
                    class_weights=fold_class_weights,
                )
            else:
                trainer = NaNAwareRegressionTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_fold,
                    eval_dataset=val_fold,
                    data_collator=data_collator,
                    callbacks=[early_stopping, best_epoch_tracker],
                )
            
            trainer.train()
            best_epochs.append(best_epoch_tracker.best_epoch)
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"Best epochs per fold: {best_epochs}")
    
    # Filter out None values (folds that didn't converge properly)
    valid_epochs = [epoch for epoch in best_epochs if epoch is not None]
    if not valid_epochs:
        # If no valid epochs found, default to 1
        logger.warning("No valid epochs found, defaulting to 1 epoch")
        optimal_epochs = 1
    else:
        optimal_epochs = int(np.round(np.mean(valid_epochs)))
    
    logger.info(f"Optimal epochs: {optimal_epochs}")
    
    return optimal_epochs


def train_final_model(
    dataset: Dataset,
    model_name: str,
    tokenizer,
    num_labels_info: Any,
    is_classification: bool,
    weighted_loss: bool,
    epochs: int,
    output_dir: Path
) -> Tuple[Any, Trainer]:
    """Train the final model using the optimal number of epochs."""
    logger.info(f"Training final model for {epochs} epochs...")
    
    # Initialize model
    if is_classification:
        if isinstance(num_labels_info, list):
            # Multi-task classification
            model = MultiTaskClassificationModel(model_name, num_labels_info)
        else:
            # Single-task classification
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels_info
            )
    else:
        # Regression
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels_info, problem_type="regression"
        )
    
    data_collator = DataCollatorWithPadding(tokenizer)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=8e-4,
        weight_decay=1e-5,
        warmup_steps=100,
        optim="schedule_free_adamw",
        lr_scheduler_type="constant",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        fp16=torch.cuda.is_available(),  # Use fp16 only on CUDA
        bf16=False,  # Disable bf16 for compatibility across environments - enable if GPU supports bf16
        save_strategy="epoch",
        eval_strategy="no",
        save_total_limit=1,
        max_grad_norm=1.0,
        logging_steps=10,
    )
    
    # Use appropriate NaN-aware trainer
    if is_classification:
        # Compute class weights for balanced loss if weighted loss is enabled
        class_weights = None
        if weighted_loss:
            labels = np.array([dataset[i]['labels'] for i in range(len(dataset))])
            class_weights = compute_class_weights(labels, is_multi_task=isinstance(num_labels_info, list))
        
        trainer = NaNAwareClassificationTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            class_weights=class_weights,
        )
    else:
        trainer = NaNAwareRegressionTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'num_labels_info': num_labels_info,
        'is_classification': is_classification,
        'is_multi_task': isinstance(num_labels_info, list),
        'epochs': epochs,
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to {output_dir}")
    
    return model, trainer


def main():
    parser = argparse.ArgumentParser(description='Fine-tune MolEncoder for molecular property prediction')
    parser.add_argument('--config', required=True, help='Path to configuration .cfg file')
    
    args = parser.parse_args()
    
    # Set reproducible seed
    set_reproducible_seed(42)
    
    # Load configuration
    config = configparser.ConfigParser()
    config.read(args.config)
    
    # Get training data file path from config
    try:
        train_data_file = config.get('DEFAULT', 'train_data_file')
    except (configparser.NoOptionError, KeyError):
        raise ValueError("train_data_file must be specified in the config file")
    
    if not train_data_file:
        raise ValueError("train_data_file cannot be empty in the config file")
    
    # Get model tar path from config
    model_tar_path = config.get('DEFAULT', 'model_tar_path', fallback='./trained_model.tar.gz')
    
    # Create temporary directory for training
    temp_output_dir = tempfile.mkdtemp(prefix='molencoder_training_')
    output_dir = Path(temp_output_dir)
    
    # Set up logging with both console and file output
    log_file = output_dir / 'training.log'
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set up new logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Force reconfiguration
    )
    
    try:
        # Get classification flag from already loaded config
        is_classification = config.getboolean('DEFAULT', 'classification', fallback=False)
        
        # Get weighted loss flag from config
        weighted_loss = config.getboolean('DEFAULT', 'weighted_loss', fallback=False)
        
        # Get debug flag from config
        debug = config.getboolean('DEFAULT', 'debug', fallback=False)
        
        logger.info(f"Task type: {'Classification' if is_classification else 'Regression'}")
        logger.info(f"Weighted loss: {'Enabled' if weighted_loss else 'Disabled'}")
        logger.info(f"Debug mode: {'Enabled' if debug else 'Disabled'}")
        
        # Load and preprocess data
        dataset, label_scaler, label_columns, num_labels_info = load_and_preprocess_data(
            train_data_file, is_classification
        )
        
        # Load tokenizer and tokenize dataset (hardcoded to MolEncoder)
        model_name = "fabikru/MolEncoder"
        logger.info(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)
        
        # Find optimal epochs or use 1 epoch in debug mode
        if debug:
            logger.info("Debug mode: Skipping cross-validation, using 1 epoch")
            optimal_epochs = 1
        else:
            optimal_epochs = find_optimal_epochs(
                tokenized_dataset, model_name, tokenizer, num_labels_info, is_classification, weighted_loss
            )
        
        # Train final model  
        final_model, final_trainer = train_final_model(
            tokenized_dataset, model_name, tokenizer, num_labels_info, 
            is_classification, weighted_loss, optimal_epochs, output_dir
        )
        
        # Save label scaler if regression
        if label_scaler is not None:
            with open(output_dir / 'label_scaler.pkl', 'wb') as f:
                pickle.dump(label_scaler, f)
            logger.info("Saved label scaler")
        
        # Save label column information
        with open(output_dir / 'label_columns.txt', 'w') as f:
            for col in label_columns:
                f.write(f"{col}\n")
        
        # Compress the trained model directory into a tar file
        
        logger.info(f"Compressing trained model to {model_tar_path}")
        
        # Ensure the parent directory of the tar file exists
        tar_path = Path(model_tar_path)
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(model_tar_path, 'w:gz') as tar:
            # Add all files from the output directory to the tar
            for file_path in output_dir.rglob('*'):
                if file_path.is_file():
                    # Use relative path within tar
                    arcname = file_path.relative_to(output_dir)
                    tar.add(file_path, arcname=arcname)
        
        # Clean up the temporary directory
        shutil.rmtree(temp_output_dir)
        
        logger.info(f"Training completed successfully! Model saved to {model_tar_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        # Clean up temporary directory on failure
        if 'temp_output_dir' in locals():
            shutil.rmtree(temp_output_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
