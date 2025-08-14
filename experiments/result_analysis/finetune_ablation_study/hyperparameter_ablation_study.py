#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import logging
import yaml
import argparse
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import optuna
from typing import List, Dict, Any
from pathlib import Path
from datasets import Dataset, DatasetDict, concatenate_datasets

from molencoder.finetune.dataset_preprocessor import preprocess_polaris_benchmark
from molencoder.finetune.label_scaler import LabelScaler
from molencoder.evaluation.metrics_from_predictions import metrics_from_predictions
from molencoder.evaluation.statistical_evaluation import repeated_measures_anova, tukey_hsd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class CustomOptimizedModelTrainer:
    """
    Custom version of OptimizedModelTrainer that works with AutoModelForSequenceClassification.
    This version only does hyperparameter optimization, not final training.
    """
    def __init__(
        self,
        base_model_name: str,
        tokenized_dataset: DatasetDict,
        datacollator: DataCollatorWithPadding,
        num_labels: int,
        optimization_db_storage_path: Path,
        n_trials: int = 50,
    ):
        self.base_model_name = base_model_name
        self.tokenized_dataset = tokenized_dataset
        self.datacollator = datacollator
        self.num_labels = num_labels
        self.optimization_db_storage_path = optimization_db_storage_path
        self.n_trials = n_trials

    def find_best_hparams(self) -> dict:
        """
        Find the best hyperparameters using Optuna optimization.
        
        Returns:
            Dictionary containing the best hyperparameters
        """
        # Define a storage string for the optimization database.
        storage_string = f"sqlite:///{self.optimization_db_storage_path}/optimization.db"

        # Create an Optuna study.
        study = optuna.create_study(
            direction="minimize",
            storage=storage_string,
        )
        # Optimize using the objective method.
        study.optimize(self.objective, n_trials=self.n_trials)
        return study.best_params

    def objective(self, trial: optuna.Trial) -> float:
        # Define the search space.
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        warmup_steps = trial.suggest_int("warmup_steps", 0, 100)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

        # Load a new instance of the model for each trial.
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name, num_labels=self.num_labels
        )

        # Use early stopping to find optimal training length
        early_stopping = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)

        # Use a temporary directory for training outputs.
        with tempfile.TemporaryDirectory() as temp_dir:
            training_args = TrainingArguments(
                num_train_epochs=100,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                optim="schedule_free_adamw",
                lr_scheduler_type="constant",
                learning_rate=learning_rate,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-8,
                weight_decay=weight_decay,
                fp16=False,
                bf16=True,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                dataloader_num_workers=8,
                dataloader_pin_memory=True,
                warmup_steps=warmup_steps,
                torch_compile=False,
                eval_on_start=False,
                output_dir=temp_dir,
                logging_dir=temp_dir,
                tf32=True 
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=self.datacollator,
                train_dataset=self.tokenized_dataset["train"],
                eval_dataset=self.tokenized_dataset["test"],
                callbacks=[early_stopping],
            )
            trainer.train()
            eval_results = trainer.evaluate()
            loss = eval_results["eval_loss"]

        return loss


def train_with_fixed_hyperparameters(
    model_name: str,
    dataset_dict: DatasetDict,
    smiles_column: str = 'smiles'
) -> Dict[str, List[float]]:
    """
    Train a model using fixed hyperparameters (same as train_and_predict).
    
    Args:
        model_name: Name or path of the pretrained model
        dataset_dict: DatasetDict containing 'train', 'validation', and 'test' splits
        smiles_column: Name of the column containing SMILES strings
        
    Returns:
        Dictionary containing the predictions and labels
    """
    # Check if dataset has the required splits
    required_splits = ["train", "validation", "test"]
    for split in required_splits:
        if split not in dataset_dict:
            raise ValueError(f"Dataset must contain a '{split}' split")
    
    required_columns = [smiles_column, "labels"]
    for split in required_splits:
        for column in required_columns:
            if column not in dataset_dict[split].features:
                raise ValueError(f"Dataset '{split}' split must contain a '{column}' column")
    
    # Tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples[smiles_column], truncation=True, max_length=500)
    dataset_dict = dataset_dict.map(tokenize_function, batched=True)
    
    # Load model and data collator
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # Regression task
    )
    
    # Train the model with fixed hyperparameters
    with tempfile.TemporaryDirectory() as temp_dir:
        training_args = TrainingArguments(
            num_train_epochs=500,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            optim="schedule_free_adamw",
            lr_scheduler_type="constant",
            learning_rate=8e-4,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            weight_decay=1e-5,
            fp16=False,
            bf16=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            warmup_steps=100,
            eval_on_start=False,
            output_dir=temp_dir,
            logging_dir=temp_dir,
            tf32=True,
            save_total_limit=2,
            torch_compile=True,
            torch_compile_backend="inductor",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            max_grad_norm=1.0,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["validation"],
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.001
            )],
        )
        trainer.train()

        combined_output = trainer.predict(dataset_dict["test"])
        predictions = combined_output.predictions.flatten().tolist()
        labels = combined_output.label_ids.flatten().tolist()
    
    return {"predictions": predictions, "labels": labels}


def train_with_optimized_hyperparameters(
    model_name: str,
    dataset_dict: DatasetDict,
    optimization_db_path: Path,
    smiles_column: str = 'smiles',
    n_trials: int = 50
) -> Dict[str, List[float]]:
    """
    Train a model using optimized hyperparameters.
    
    Args:
        model_name: Name or path of the pretrained model
        dataset_dict: DatasetDict containing 'train', 'validation', and 'test' splits
        optimization_db_path: Path to store optimization database
        smiles_column: Name of the column containing SMILES strings
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary containing the predictions and labels
    """
    # Check if dataset has the required splits
    required_splits = ["train", "validation", "test"]
    for split in required_splits:
        if split not in dataset_dict:
            raise ValueError(f"Dataset must contain a '{split}' split")
    
    required_columns = [smiles_column, "labels"]
    for split in required_splits:
        for column in required_columns:
            if column not in dataset_dict[split].features:
                raise ValueError(f"Dataset '{split}' split must contain a '{column}' column")
    
    # Tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples[smiles_column], truncation=True, max_length=500)
    tokenized_dataset = dataset_dict.map(tokenize_function, batched=True)
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    
    # Use CustomOptimizedModelTrainer for hyperparameter optimization
    # FAIR COMPARISON: Use same train/validation split as fixed hyperparameters
    # Train on 'train', validate on 'validation', optimize hyperparameters
    # Then use optimized hyperparameters to train final model and predict on 'test'
    optimization_dataset = DatasetDict({
        'train': tokenized_dataset['train'],
        'test': tokenized_dataset['validation']  # Use validation for hyperparameter optimization
    })
    
    trainer = CustomOptimizedModelTrainer(
        base_model_name=model_name,
        tokenized_dataset=optimization_dataset,
        datacollator=data_collator,
        num_labels=1,
        optimization_db_storage_path=optimization_db_path,
        n_trials=n_trials
    )
    
    # Get optimized hyperparameters (but don't use the trained model)
    optimized_hparams = trainer.find_best_hparams()
    
    # Now train a fresh model with optimized hyperparameters on the same data as fixed approach
    # Train on 'train', validate on 'validation', predict on 'test'
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
    )
    
    # Use early stopping for final training (same as fixed approach)
    with tempfile.TemporaryDirectory() as temp_dir:
        training_args = TrainingArguments(
            num_train_epochs=500,  # Set high, early stopping will determine actual epochs
            per_device_train_batch_size=optimized_hparams["batch_size"],
            per_device_eval_batch_size=optimized_hparams["batch_size"],
            optim="schedule_free_adamw",
            lr_scheduler_type="constant",
            learning_rate=optimized_hparams["learning_rate"],
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            weight_decay=optimized_hparams["weight_decay"],
            fp16=False,
            bf16=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            warmup_steps=optimized_hparams["warmup_steps"],
            eval_on_start=False,
            output_dir=temp_dir,
            logging_dir=temp_dir,
            tf32=True,
            save_total_limit=2,
            torch_compile=True,
            torch_compile_backend="inductor",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            max_grad_norm=1.0,
        )
        
        # Train on same data as fixed approach: train for training, validation for early stopping
        final_trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.001
            )],
        )
        final_trainer.train()
        
        # Predict on test set (same as fixed approach)
        combined_output = final_trainer.predict(tokenized_dataset["test"])
        predictions = combined_output.predictions.flatten().tolist()
        labels = combined_output.label_ids.flatten().tolist()
    
    return {"predictions": predictions, "labels": labels}


def cross_validate_hyperparameter_methods(
    model_name: str,
    dataset: Dataset,
    optimization_db_path: Path,
    smiles_column: str = 'smiles',
    n_repeats: int = 5,
    n_folds: int = 5,
    n_trials: int = 50,
    random_state: int = 42
) -> Dict[str, List[Dict[str, float]]]:
    """
    Perform 5x5 cross-validation comparing fixed vs optimized hyperparameters.
    
    Args:
        model_name: Name of the model to evaluate
        dataset: Hugging Face dataset with 'smiles' and 'labels' columns
        optimization_db_path: Path to store optimization databases
        smiles_column: Name of the SMILES column to use
        n_repeats: Number of cross-validation repeats
        n_folds: Number of folds for cross-validation
        n_trials: Number of optimization trials
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with method names as keys and lists of metrics for each fold as values
    """
    # Initialize results dictionary
    results = {
        'Fixed_Hyperparameters': [],
        'Optimized_Hyperparameters': []
    }
    
    # Check if dataset has required columns
    if 'smiles' not in dataset.features or 'labels' not in dataset.features:
        raise ValueError("Dataset must contain 'smiles' and 'labels' columns")
    
    if smiles_column != 'smiles' and smiles_column not in dataset.features:
        raise ValueError(f"Dataset must contain '{smiles_column}' column")
    
    # Get dataset size and calculate fold size
    dataset_size = len(dataset)
    fold_size = dataset_size // n_folds
    
    # Perform n_repeats of n_folds cross-validation
    for repeat in range(n_repeats):
        logging.info(f"Repeat {repeat+1}/{n_repeats}")
        
        # Create a new random seed for each repeat
        current_seed = random_state + repeat
        
        # Shuffle the dataset
        shuffled_dataset = dataset.shuffle(seed=current_seed)
        
        # Create folds
        folds = []
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else dataset_size
            fold = shuffled_dataset.select(range(start_idx, end_idx))
            folds.append(fold)
        
        # Cycle through the folds
        for fold_idx in range(n_folds):
            logging.info(f"  Fold {fold_idx+1}/{n_folds}")
            
            # Determine which folds to use for train, validation, and test
            test_fold_idx = fold_idx
            val_fold_idx = (fold_idx + 1) % n_folds
            train_fold_indices = [(fold_idx + i) % n_folds for i in range(2, 5)]
            
            # Get train, validation, and test datasets
            train_folds = [folds[i] for i in train_fold_indices]
            val_fold = folds[val_fold_idx]
            test_fold = folds[test_fold_idx]
            
            # Concatenate train folds
            train_dataset = concatenate_datasets(train_folds)
            
            # Create dataset dictionary
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
            
            # Create optimization database path for this fold
            fold_db_path = optimization_db_path / f"repeat_{repeat}_fold_{fold_idx}"
            fold_db_path.mkdir(parents=True, exist_ok=True)
            
            # FAIR COMPARISON APPROACH:
            # Both methods use the same data splits:
            # - Train on 'train' fold
            # - Use 'validation' fold for early stopping/hyperparameter optimization
            # - Evaluate on 'test' fold
            # The only difference is:
            # - Fixed: Uses predetermined hyperparameters
            # - Optimized: Uses Optuna to find best hyperparameters on train/validation, 
            #   then trains a fresh model with those hyperparameters
            
            # Evaluate fixed hyperparameters
            try:
                logging.info(f"    Training with fixed hyperparameters")
                fixed_predictions = train_with_fixed_hyperparameters(
                    model_name, dataset_dict, smiles_column
                )
                
                # Rescale predictions
                rescaled_predictions = label_scaler.scale_predictions(fixed_predictions['predictions'])
                
                # Calculate metrics
                rescaled_predictions_dict = {
                    'predictions': rescaled_predictions,
                    'labels': dataset_dict['test']['labels']
                }
                fixed_metrics = metrics_from_predictions(rescaled_predictions_dict)
                results['Fixed_Hyperparameters'].append(fixed_metrics)
                
                logging.info(f"      Fixed hyperparameters metrics:")
                for metric_name, value in fixed_metrics.items():
                    logging.info(f"        {metric_name}: {value:.4f}")
                    
            except Exception as e:
                logging.error(f"Fixed hyperparameters failed on repeat {repeat+1}, fold {fold_idx+1}: {str(e)}")
                continue
            
            # Evaluate optimized hyperparameters
            try:
                logging.info(f"    Training with optimized hyperparameters")
                optimized_predictions = train_with_optimized_hyperparameters(
                    model_name, dataset_dict, fold_db_path, smiles_column, n_trials
                )
                
                # Rescale predictions
                rescaled_predictions = label_scaler.scale_predictions(optimized_predictions['predictions'])
                
                # Calculate metrics
                rescaled_predictions_dict = {
                    'predictions': rescaled_predictions,
                    'labels': dataset_dict['test']['labels']
                }
                optimized_metrics = metrics_from_predictions(rescaled_predictions_dict)
                results['Optimized_Hyperparameters'].append(optimized_metrics)
                
                logging.info(f"      Optimized hyperparameters metrics:")
                for metric_name, value in optimized_metrics.items():
                    logging.info(f"        {metric_name}: {value:.4f}")
                    
            except Exception as e:
                logging.error(f"Optimized hyperparameters failed on repeat {repeat+1}, fold {fold_idx+1}: {str(e)}")
                continue
    
    return results


def reshape_results_to_dataframe(results: Dict[str, List[Dict[str, float]]]) -> pd.DataFrame:
    """
    Reshape the results into a pandas DataFrame.
    
    Args:
        results: Dictionary with method names as keys and lists of metrics for each fold as values
        
    Returns:
        DataFrame with columns: method, fold, metric_name, value
    """
    rows = []
    
    for method_name, fold_metrics in results.items():
        for fold_idx, metrics in enumerate(fold_metrics):
            for metric_name, value in metrics.items():
                rows.append({
                    'method': method_name,
                    'fold': fold_idx,
                    'metric_name': metric_name,
                    'value': value
                })
    
    return pd.DataFrame(rows)


def load_and_prepare_data_for_analysis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for statistical analysis.
    
    Args:
        df: DataFrame with columns: method, fold, metric_name, value
        
    Returns:
        Dictionary with metric names as keys and prepared DataFrames as values
    """
    prepared_data = {}
    
    # Get unique metrics
    metrics = df['metric_name'].unique()
    
    for metric in metrics:
        # Filter data for current metric
        metric_data = df[df['metric_name'] == metric]
        
        # Pivot the data to have methods as columns and folds as rows
        pivot_data = metric_data.pivot(
            index='fold',
            columns='method',
            values='value'
        )
        
        prepared_data[metric] = pivot_data
    
    return prepared_data


def find_best_method(data: pd.DataFrame, metric_name: str) -> str:
    """
    Find the best performing method based on the metric values.
    """
    # Calculate mean performance for each method
    method_means = data.mean()
    
    if metric_name in ['mae', 'mse']:
        # For MAE and MSE, lower is better
        best_method = method_means.idxmin()
    else:
        # For R2 and rho, higher is better
        best_method = method_means.idxmax()
    
    return best_method


def analyze_and_plot_results(results_df: pd.DataFrame, dataset_name: str, output_dir: Path):
    """
    Analyze results and create plots similar to the example script.
    
    Args:
        results_df: DataFrame with results
        dataset_name: Name of the dataset
        output_dir: Directory to save plots
    """
    # Prepare data for analysis
    prepared_data = load_and_prepare_data_for_analysis(results_df)
    
    # Create output directory for plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for metric, metric_data in prepared_data.items():
        logging.info(f"Analyzing {metric} for {dataset_name}")
        
        # Perform repeated measures ANOVA
        anova_results = repeated_measures_anova(metric_data)
        
        # Check if the ANOVA result is significant (p < 0.05)
        if anova_results['p_value'] < 0.05:
            logging.info(f"Significant differences found for {metric} (p = {anova_results['p_value']:.4f})")
            
            # Find the best performing method
            best_method = find_best_method(metric_data, metric)
            logging.info(f"Best performing method for {metric}: {best_method}")
            
            # Perform Tukey's HSD test
            tukey_results = tukey_hsd(metric_data)
            
            # Create and save the plot
            plt.figure(figsize=(10, 6))
            tukey_results['results'].plot_simultaneous(comparison_name=best_method)
            plt.title(f"Tukey HSD Test Results for {metric.upper()} - {dataset_name}")
            plt.tight_layout()
            
            # Save plot
            plot_path = plots_dir / f"{dataset_name}_{metric}_tukey_hsd.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        else:
            logging.info(f"No significant differences found for {metric} (p = {anova_results['p_value']:.4f})")


def main(config_path: Path):
    """
    Run the hyperparameter ablation study.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logging.info("Starting hyperparameter ablation study...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config['model_name']
    dataset_names = config['dataset_names']
    results_folder = Path(config['results_folder'])
    n_trials = config.get('n_trials', 50)
    
    # Create results folder if it doesn't exist
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # Create optimization database folder
    optimization_db_folder = results_folder / "optimization_dbs"
    optimization_db_folder.mkdir(parents=True, exist_ok=True)
    
    # Save a copy of the config file in the results folder
    config_copy_path = results_folder / 'config.yaml'
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f)
    
    # Process each dataset
    for dataset_name in dataset_names:
        logger.info(f"Processing dataset: {dataset_name}")
        
        try:
            # Load and preprocess dataset
            dataset = preprocess_polaris_benchmark(dataset_name, add_explicit_hydrogens=False)
            logger.info(f"Successfully loaded dataset: {dataset_name}")
            
            # Create optimization database path for this dataset
            dataset_db_path = optimization_db_folder / dataset_name.replace('/', '_')
            
            # Perform cross-validation
            results = cross_validate_hyperparameter_methods(
                model_name=model_name,
                dataset=dataset,
                optimization_db_path=dataset_db_path,
                smiles_column='smiles',
                n_repeats=5,
                n_folds=5,
                n_trials=n_trials
            )
            
            # Reshape results to DataFrame
            results_df = reshape_results_to_dataframe(results)
            
            # Save results to CSV
            output_file = results_folder / f"{dataset_name.split('/')[-1]}_hyperparameter_ablation_results.csv"
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            
            # Analyze and plot results
            analyze_and_plot_results(results_df, dataset_name.split('/')[-1], results_folder)
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {e}")

    logging.info("Hyperparameter ablation study completed.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run hyperparameter ablation study.')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to the YAML configuration file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)
    
    # Run the main function
    main(config_path) 