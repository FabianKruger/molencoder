#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import logging
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from pathlib import Path
from datasets import Dataset, DatasetDict, concatenate_datasets

from molencoder.finetune.dataset_preprocessor import preprocess_polaris_benchmark
from molencoder.finetune.label_scaler import LabelScaler
from molencoder.evaluation.metrics_from_predictions import metrics_from_predictions
from molencoder.evaluation.statistical_evaluation import repeated_measures_anova, tukey_hsd

# Import baseline implementations
from molencoder.evaluation.baselines.ecfp4_xgboost_baseline import ECFP4XGBoostBaseline
from molencoder.evaluation.baselines.transformer_baseline import TransformerBaseline

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def cross_validate_mixed_models(
    transformer_models: List[str],
    dataset: Dataset,
    n_repeats: int = 5,
    n_folds: int = 5,
    random_state: int = 42,
    include_ecfp4_baseline: bool = True
) -> Optional[Dict[str, List[Dict[str, float]]]]:
    """
    Perform 5x5 cross-validation on mixed model types (transformers + ECFP4+XGBoost).
    
    FAIL-FAST BEHAVIOR: If any model fails on any fold, the entire cross-validation
    for this dataset is aborted and None is returned. This ensures that comparisons
    are only made when all models successfully complete all folds.
    
    Args:
        transformer_models: List of transformer model names
        dataset: Hugging Face dataset with 'smiles' and 'labels' columns
        n_repeats: Number of cross-validation repeats
        n_folds: Number of folds for cross-validation
        random_state: Random seed for reproducibility
        include_ecfp4_baseline: Whether to include ECFP4+XGBoost baseline
        
    Returns:
        Dictionary with model names as keys and lists of metrics for each fold as values,
        or None if any model failed on any fold
    """
    # Initialize results dictionary
    all_models = transformer_models.copy()
    if include_ecfp4_baseline:
        all_models.append("ECFP4+XGBoost")
    
    results = {model: [] for model in all_models}
    
    # Check if dataset has required columns
    if 'smiles' not in dataset.features or 'labels' not in dataset.features:
        raise ValueError("Dataset must contain 'smiles' and 'labels' columns")
    
    # Get dataset size and calculate fold size
    dataset_size = len(dataset)
    fold_size = dataset_size // n_folds
    
    # Initialize baseline models
    ecfp4_baseline = ECFP4XGBoostBaseline() if include_ecfp4_baseline else None
    transformer_baseline = TransformerBaseline()
    
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
            
            # Scale train and validation labels for transformer models
            scaled_train_dataset = label_scaler.scale_labels(train_dataset)
            scaled_val_dataset = label_scaler.scale_labels(val_fold)
            
            # Update dataset dictionary with scaled datasets for transformers
            scaled_dataset_dict = DatasetDict({
                'train': scaled_train_dataset,
                'validation': scaled_val_dataset,
                'test': test_fold  # Keep test unscaled for consistent evaluation
            })
            
            # Evaluate transformer models - FAIL FAST if any model fails
            for model_name in transformer_models:
                metrics = _evaluate_transformer_model_safely(
                    model_name, scaled_dataset_dict, label_scaler, 
                    repeat, fold_idx, transformer_baseline
                )
                if metrics is not None:
                    results[model_name].append(metrics)
                else:
                    logging.error(f"Model {model_name} failed on repeat {repeat+1}, fold {fold_idx+1}. "
                                f"Aborting cross-validation for this dataset.")
                    return None  # FAIL FAST - abort entire dataset
            
            # Evaluate ECFP4+XGBoost baseline - FAIL FAST if it fails
            if include_ecfp4_baseline:
                metrics = _evaluate_ecfp4_model_safely(
                    dataset_dict, repeat, fold_idx, ecfp4_baseline
                )
                if metrics is not None:
                    results["ECFP4+XGBoost"].append(metrics)
                else:
                    logging.error(f"ECFP4+XGBoost baseline failed on repeat {repeat+1}, fold {fold_idx+1}. "
                                f"Aborting cross-validation for this dataset.")
                    return None  # FAIL FAST - abort entire dataset
    
    logging.info("Cross-validation completed successfully for all models.")
    return results


def _evaluate_transformer_model_safely(model_name, dataset_dict, label_scaler, repeat, fold_idx, transformer_baseline):
    """Safely evaluate a transformer model with error handling."""
    try:
        return _evaluate_transformer_model(model_name, dataset_dict, label_scaler, repeat, fold_idx, transformer_baseline)
    except Exception as e:
        logging.warning(f"Transformer model {model_name} failed on repeat {repeat+1}, fold {fold_idx+1}: {str(e)}")
        return None


def _evaluate_transformer_model(model_name, dataset_dict, label_scaler, repeat, fold_idx, transformer_baseline):
    """Evaluate a single transformer model."""
    logging.info(f"    Training transformer model: {model_name}")
    
    # Train model and get predictions
    predictions_dict = transformer_baseline.train_and_predict(model_name, dataset_dict)
    
    # Check for NaN predictions
    predictions_array = np.array(predictions_dict['predictions'])
    if np.any(np.isnan(predictions_array)) or np.any(np.isinf(predictions_array)):
        nan_count = np.sum(np.isnan(predictions_array))
        inf_count = np.sum(np.isinf(predictions_array))
        raise ValueError(f"Model produced {nan_count} NaN predictions and {inf_count} infinite predictions")
    
    # Rescale predictions
    rescaled_predictions = label_scaler.scale_predictions(predictions_dict['predictions'])
    
    # Check for NaN predictions after rescaling
    rescaled_array = np.array(rescaled_predictions)
    if np.any(np.isnan(rescaled_array)) or np.any(np.isinf(rescaled_array)):
        nan_count = np.sum(np.isnan(rescaled_array))
        inf_count = np.sum(np.isinf(rescaled_array))
        raise ValueError(f"Rescaling produced {nan_count} NaN predictions and {inf_count} infinite predictions")
    
    # Create dictionary with rescaled predictions and original test labels
    rescaled_predictions_dict = {
        'predictions': rescaled_predictions,
        'labels': dataset_dict['test']['labels']
    }
    
    # Calculate metrics
    metrics = metrics_from_predictions(rescaled_predictions_dict)
    
    # Check if any metrics are NaN
    for metric_name, value in metrics.items():
        if np.isnan(value) or np.isinf(value):
            raise ValueError(f"Metric '{metric_name}' is {value}")
    
    # Print metrics
    logging.info(f"      Metrics for {model_name} (Repeat {repeat+1}, Fold {fold_idx+1}):")
    for metric_name, value in metrics.items():
        logging.info(f"        {metric_name}: {value:.4f}")
    
    return metrics


def _evaluate_ecfp4_model_safely(dataset_dict, repeat, fold_idx, ecfp4_baseline):
    """Safely evaluate ECFP4+XGBoost baseline with error handling."""
    try:
        return _evaluate_ecfp4_model(dataset_dict, repeat, fold_idx, ecfp4_baseline)
    except Exception as e:
        logging.warning(f"ECFP4+XGBoost baseline failed on repeat {repeat+1}, fold {fold_idx+1}: {str(e)}")
        return None


def _evaluate_ecfp4_model(dataset_dict, repeat, fold_idx, ecfp4_baseline):
    """Evaluate ECFP4+XGBoost baseline."""
    logging.info(f"    Training ECFP4+XGBoost baseline")
    
    # Train model and get predictions (no label scaling needed for XGBoost)
    predictions_dict = ecfp4_baseline.train_and_predict(dataset_dict)
    
    # Check for NaN predictions
    predictions_array = np.array(predictions_dict['predictions'])
    if np.any(np.isnan(predictions_array)) or np.any(np.isinf(predictions_array)):
        nan_count = np.sum(np.isnan(predictions_array))
        inf_count = np.sum(np.isinf(predictions_array))
        raise ValueError(f"ECFP4+XGBoost produced {nan_count} NaN predictions and {inf_count} infinite predictions")
    
    # Calculate metrics (no rescaling needed)
    metrics = metrics_from_predictions(predictions_dict)
    
    # Check if any metrics are NaN
    for metric_name, value in metrics.items():
        if np.isnan(value) or np.isinf(value):
            raise ValueError(f"Metric '{metric_name}' is {value}")
    
    # Print metrics
    logging.info(f"      Metrics for ECFP4+XGBoost (Repeat {repeat+1}, Fold {fold_idx+1}):")
    for metric_name, value in metrics.items():
        logging.info(f"        {metric_name}: {value:.4f}")
    
    return metrics


def reshape_results_to_dataframe(results: Dict[str, List[Dict[str, float]]]) -> pd.DataFrame:
    """
    Reshape the results from cross_validate_mixed_models into a pandas DataFrame.
    
    Args:
        results: Dictionary with model names as keys and lists of metrics for each fold as values
        
    Returns:
        DataFrame with columns: model, fold, metric_name, value
    """
    rows = []
    
    for model_name, fold_metrics in results.items():
        for fold_idx, metrics in enumerate(fold_metrics):
            for metric_name, value in metrics.items():
                rows.append({
                    'model': model_name,
                    'fold': fold_idx,
                    'metric_name': metric_name,
                    'value': value
                })
    
    return pd.DataFrame(rows)


def load_and_prepare_data_for_analysis(results_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for statistical analysis.
    
    Args:
        results_df: DataFrame with results
        
    Returns:
        Dictionary with metric names as keys and prepared DataFrames as values
    """
    prepared_data = {}
    
    # Get unique metrics
    metrics = results_df['metric_name'].unique()
    
    for metric in metrics:
        # Filter data for current metric
        metric_data = results_df[results_df['metric_name'] == metric]
        
        # Pivot the data to have models as columns and folds as rows
        pivot_data = metric_data.pivot(
            index='fold',
            columns='model',
            values='value'
        )
        
        prepared_data[metric] = pivot_data
    
    return prepared_data


def find_best_model(data: pd.DataFrame, metric_name: str) -> str:
    """
    Find the best performing model based on the metric values.
    """
    # Calculate mean performance for each model
    model_means = data.mean()
    
    if metric_name in ['mae', 'mse']:
        # For MAE and MSE, lower is better
        best_model = model_means.idxmin()
    else:
        # For R2 and rho, higher is better
        best_model = model_means.idxmax()
    
    return best_model


def analyze_and_plot_results(results_df: pd.DataFrame, dataset_name: str, output_dir: Path):
    """
    Analyze results and create plots with improved aesthetics matching the experiment plotting style.
    
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
    
    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*80}")
    
    for metric, metric_data in prepared_data.items():
        print(f"\nMetric: {metric.upper()}")
        print("-" * 50)
        
        # Calculate descriptive statistics
        for model in metric_data.columns:
            mean_val = metric_data[model].mean()
            std_val = metric_data[model].std()
            print(f"  {model}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Perform repeated measures ANOVA
        anova_results = repeated_measures_anova(metric_data)
        print(f"  Repeated Measures ANOVA: F={anova_results['f_statistic']:.4f}, p={anova_results['p_value']:.4f}")
        
        # Determine significance
        if anova_results['p_value'] < 0.001:
            significance = "***"
        elif anova_results['p_value'] < 0.01:
            significance = "**"
        elif anova_results['p_value'] < 0.05:
            significance = "*"
        else:
            significance = "ns"
        
        print(f"  Significance: {significance}")
        
        # Find the best performing model
        best_model = find_best_model(metric_data, metric)
        print(f"  Best performing model: {best_model}")
        
        # Always perform Tukey's HSD test and create plot with improved aesthetics
        tukey_results = tukey_hsd(metric_data)
        
        # Create and save the Tukey HSD plot with improved aesthetics matching experiment_plotting style
        fig, ax = plt.subplots(figsize=(7.5, 2.5))
        tukey_results['results'].plot_simultaneous(comparison_name=best_model, ax=ax)
        
        # Apply aesthetic styling to match experiment_plotting functions
        ax.set_title(dataset_name, fontsize=10, color='#666666')
        ax.set_xlabel(metric.upper(), fontsize=10, color='#666666')
        ax.set_ylabel('Model', fontsize=10, color='#666666')
        
        # Set tick parameters to match experiment_plotting style
        ax.tick_params(axis='x', labelsize=8, colors='#666666')
        ax.tick_params(axis='y', labelsize=8, colors='#333333')
        
        # Make borders light grey to match experiment_plotting style
        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')
            spine.set_linewidth(0.8)
        
        # Remove lowest and highest y-axis ticks if there are more than 2
        yticks = ax.get_yticks()
        if len(yticks) > 2:
            new_yticks = yticks[1:-1]
            ax.set_yticks(new_yticks)
        
        plt.tight_layout()
        
        # Save the plot in both PNG and PDF formats to match experiment_plotting style
        base_filename = f"{dataset_name}_{metric}_model_comparison"
        
        # Save as PNG
        png_filename = f"{base_filename}.png"
        plt.savefig(plots_dir / png_filename, dpi=300, bbox_inches='tight')
        print(f"  Model comparison plot saved: {png_filename}")
        
        # Save as PDF
        pdf_filename = f"{base_filename}.pdf"
        plt.savefig(plots_dir / pdf_filename, bbox_inches='tight')
        print(f"  Model comparison plot saved: {pdf_filename}")
        
        plt.close()
        
        # Print Tukey HSD summary
        print(f"  Tukey HSD Summary:")
        print(f"    {tukey_results['summary_table']}")
        
        # Check if the ANOVA result is significant (p < 0.05)
        if anova_results['p_value'] < 0.05:
            print(f"  Significant differences found for {metric}")
        else:
            print(f"  No significant differences found for {metric}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: Path):
    """
    Run model comparison study.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logging.info("Starting model comparison study...")
    
    # Load configuration
    config = load_config(config_path)
    dataset_names = config['dataset_names']
    transformer_models = config['transformer_models']
    include_ecfp4_baseline = config.get('include_ecfp4_baseline', True)
    
    results_folder = Path(config['results_folder'])
    
    # Create results folder if it doesn't exist
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # Save a copy of the config file in the results folder
    config_copy_path = results_folder / 'config.yaml'
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f)
    
    # Load all datasets
    datasets = {}
    for dataset_name in dataset_names:
        logger.info(f"Loading dataset: {dataset_name}")
        try:
            datasets[dataset_name] = preprocess_polaris_benchmark(
                dataset_name, 
                add_explicit_hydrogens=False  # Not needed for this comparison
            )
            logger.info(f"Successfully loaded dataset: {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            continue
    
    logging.info("Finished loading and preprocessing datasets.")
    
    # Evaluate models on each dataset
    for dataset_name, dataset in datasets.items():
        logger.info(f"Evaluating models on dataset: {dataset_name}")
        
        try:
            # Perform cross-validation
            results = cross_validate_mixed_models(
                transformer_models=transformer_models,
                dataset=dataset,
                n_repeats=5,
                n_folds=5,
                include_ecfp4_baseline=include_ecfp4_baseline
            )
            
            # Check if cross-validation failed (any model failed on any fold)
            if results is None:
                logger.error(f"Cross-validation failed for dataset {dataset_name}. "
                           f"One or more models failed during evaluation. Skipping this dataset.")
                continue
            
            # Reshape results to DataFrame
            results_df = reshape_results_to_dataframe(results)
            
            # Save results to CSV
            output_file = results_folder / f"{dataset_name.split('/')[-1]}_comparison_results.csv"
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            
            # Perform statistical analysis and create plots
            analyze_and_plot_results(results_df, dataset_name.split('/')[-1], results_folder)
            
        except Exception as e:
            logger.error(f"Error evaluating models on dataset {dataset_name}: {e}")

    logging.info("Model comparison study completed.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run model comparison study with configuration from YAML file.')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to the YAML configuration file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)
    
    # Run the main function
    main(config_path) 