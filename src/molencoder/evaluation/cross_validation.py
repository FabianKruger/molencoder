import numpy as np
import logging
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict, concatenate_datasets
from molencoder.finetune.label_scaler import LabelScaler
from molencoder.finetune.train_and_predict import train_and_predict
from molencoder.evaluation.metrics_from_predictions import metrics_from_predictions


def cross_validate_models(
    models: List[str],
    dataset: Dataset,
    models_with_explicit_hydrogens: List[str] = None,
    n_repeats: int = 5,
    n_folds: int = 5,
    random_state: int = 42
) -> Dict[str, List[Dict[str, float]]]:
    """
    Perform 5x5 cross-validation on models using a dataset.
    Handles two types of models: those that use standard SMILES and 
    those that need explicit hydrogens.
    
    Models that produce NaN predictions or encounter other errors will be skipped
    with a warning logged, allowing the cross-validation to continue with successful models.
    
    For each repeat:
    - Shuffle the dataset using its built-in shuffle method
    - Split into 5 equal parts using select
    - Cycle through the folds, using 3 for training, 1 for validation, 1 for testing
    - Scale labels using LabelScaler
    - Train and evaluate each model with the appropriate SMILES column
    - Rescale predictions and calculate metrics
    
    Args:
        models: List of model names that use standard SMILES
        dataset: Hugging Face dataset with 'smiles' and 'labels' columns
        models_with_explicit_hydrogens: List of model names that need explicit hydrogens
        n_repeats: Number of cross-validation repeats (default: 5)
        n_folds: Number of folds for cross-validation (default: 5)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        Dictionary with model names as keys and lists of metrics for each fold as values.
        Models that failed will have empty lists.
    """
    if models_with_explicit_hydrogens is None:
        models_with_explicit_hydrogens = []
    
    all_models = models + models_with_explicit_hydrogens
    
    results = {model: [] for model in all_models}
    
    failed_models = set()
    
    if 'smiles' not in dataset.features or 'labels' not in dataset.features:
        raise ValueError("Dataset must contain 'smiles' and 'labels' columns")
    
    if models_with_explicit_hydrogens and 'smiles_with_hydrogens' not in dataset.features:
        raise ValueError("Models require 'smiles_with_hydrogens' column, but it's not in the dataset")
    
    dataset_size = len(dataset)
    fold_size = dataset_size // n_folds
    
    # Perform n_repeats of n_folds cross-validation
    for repeat in range(n_repeats):
        logging.info(f"Repeat {repeat+1}/{n_repeats}")
        
        # Create a new random seed for each repeat
        current_seed = random_state + repeat
        
        # Shuffle the dataset using its built-in shuffle method
        shuffled_dataset = dataset.shuffle(seed=current_seed)
        
        # Create folds using select
        folds = []
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else dataset_size
            fold = shuffled_dataset.select(range(start_idx, end_idx))
            folds.append(fold)
        
        # Cycle through the folds for each fold
        for fold_idx in range(n_folds):
            logging.info(f"  Fold {fold_idx+1}/{n_folds}")
            
            # Determine which folds to use for train, validation, and test
            # Use 3 folds for training, 1 for validation, 1 for testing
            test_fold_idx = fold_idx
            val_fold_idx = (fold_idx + 1) % n_folds
            train_fold_indices = [(fold_idx + i) % n_folds for i in range(2, 5)]
            
            # Get train, validation, and test datasets
            train_folds = [folds[i] for i in train_fold_indices]
            val_fold = folds[val_fold_idx]
            test_fold = folds[test_fold_idx]
            
            # Concatenate train folds
            train_dataset = concatenate_datasets(train_folds)
            
            # Create dataset dictionary for train_and_predict
            dataset_dict = DatasetDict({
                'train': train_dataset,
                'validation': val_fold,
                'test': test_fold
            })
            
            # Create LabelScaler from training data
            label_scaler = LabelScaler(train_dataset)
            
            # Scale train and validation labels
            scaled_train_dataset = label_scaler.scale_labels(train_dataset)
            scaled_val_dataset = label_scaler.scale_labels(val_fold)
            
            # Update dataset dictionary with scaled datasets
            dataset_dict['train'] = scaled_train_dataset
            dataset_dict['validation'] = scaled_val_dataset
            
            # Train and evaluate each standard model (using 'smiles' column)
            for model_name in models:
                if model_name not in failed_models:
                    metrics = _evaluate_model_safely(model_name, dataset_dict, label_scaler, repeat, fold_idx, smiles_column='smiles')
                    if metrics is not None:
                        results[model_name].append(metrics)
                    else:
                        failed_models.add(model_name)
                        logging.warning(f"Model {model_name} failed and will be skipped for remaining folds")
            
            # Train and evaluate each model that needs explicit hydrogens (using 'smiles_with_hydrogens' column)
            for model_name in models_with_explicit_hydrogens:
                if model_name not in failed_models:
                    metrics = _evaluate_model_safely(model_name, dataset_dict, label_scaler, repeat, fold_idx, smiles_column='smiles_with_hydrogens')
                    if metrics is not None:
                        results[model_name].append(metrics)
                    else:
                        failed_models.add(model_name)
                        logging.warning(f"Model {model_name} failed and will be skipped for remaining folds")
    
    # Log summary of failed models
    if failed_models:
        logging.warning(f"The following models failed and were skipped: {list(failed_models)}")
    
    return results


def _evaluate_model_safely(model_name, dataset_dict, label_scaler, repeat, fold_idx, smiles_column='smiles'):
    """
    Safely train and evaluate a single model with error handling.
    
    Args:
        model_name: Name of the model to evaluate
        dataset_dict: Dataset dictionary with train, validation, and test datasets
        label_scaler: LabelScaler instance for rescaling predictions
        repeat: Current repeat number
        fold_idx: Current fold index
        smiles_column: Name of the SMILES column to use ('smiles' or 'smiles_with_hydrogens')
        
    Returns:
        Dictionary of evaluation metrics, or None if the model failed
    """
    try:
        return _evaluate_model(model_name, dataset_dict, label_scaler, repeat, fold_idx, smiles_column)
    except Exception as e:
        logging.warning(f"Model {model_name} failed on repeat {repeat+1}, fold {fold_idx+1}: {str(e)}")
        return None


def _evaluate_model(model_name, dataset_dict, label_scaler, repeat, fold_idx, smiles_column='smiles'):
    """
    Train and evaluate a single model.
    
    Args:
        model_name: Name of the model to evaluate
        dataset_dict: Dataset dictionary with train, validation, and test datasets
        label_scaler: LabelScaler instance for rescaling predictions
        repeat: Current repeat number
        fold_idx: Current fold index
        smiles_column: Name of the SMILES column to use ('smiles' or 'smiles_with_hydrogens')
        
    Returns:
        Dictionary of evaluation metrics
    """
    logging.info(f"    Training model: {model_name}")
    
    predictions_dict = train_and_predict(model_name, dataset_dict, smiles_column=smiles_column)
    
    predictions_array = np.array(predictions_dict['predictions'])
    if np.any(np.isnan(predictions_array)) or np.any(np.isinf(predictions_array)):
        nan_count = np.sum(np.isnan(predictions_array))
        inf_count = np.sum(np.isinf(predictions_array))
        raise ValueError(f"Model produced {nan_count} NaN predictions and {inf_count} infinite predictions")
    
    rescaled_predictions = label_scaler.scale_predictions(predictions_dict['predictions'])
    
    rescaled_array = np.array(rescaled_predictions)
    if np.any(np.isnan(rescaled_array)) or np.any(np.isinf(rescaled_array)):
        nan_count = np.sum(np.isnan(rescaled_array))
        inf_count = np.sum(np.isinf(rescaled_array))
        raise ValueError(f"Rescaling produced {nan_count} NaN predictions and {inf_count} infinite predictions")
    
    rescaled_predictions_dict = {
        'predictions': rescaled_predictions,
        'labels': dataset_dict['test']['labels']
    }
    
    metrics = metrics_from_predictions(rescaled_predictions_dict)
    
    for metric_name, value in metrics.items():
        if np.isnan(value) or np.isinf(value):
            raise ValueError(f"Metric '{metric_name}' is {value}")
    
    # Print metrics
    logging.info(f"      Metrics for {model_name} (Repeat {repeat+1}, Fold {fold_idx+1}):")
    for metric_name, value in metrics.items():
        logging.info(f"        {metric_name}: {value:.4f}")
    
    return metrics  