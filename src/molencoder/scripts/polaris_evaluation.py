#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import logging
import yaml
import argparse
from typing import List, Dict, Any
from pathlib import Path

from molencoder.finetune.dataset_preprocessor import preprocess_polaris_benchmark
from molencoder.evaluation.cross_validation import cross_validate_models
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def reshape_results_to_dataframe(results: Dict[str, List[Dict[str, float]]]) -> pd.DataFrame:
    """
    Reshape the results from cross_validate_models into a pandas DataFrame.
    
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
    Evaluate models on multiple datasets using cross-validation.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    # Configure logging
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logging.info("Starting Polaris evaluation...")
    
    # Load configuration
    config = load_config(config_path)
    dataset_names = config['dataset_names']
    
    # Get model lists
    model_names = config.get('model_names', [])
    models_with_explicit_hydrogens = config.get('models_with_explicit_hydrogens', [])

    if models_with_explicit_hydrogens is None:
        models_with_explicit_hydrogens = [] 

    if model_names is None:
        model_names = []

    # Ensure model lists don't have duplicates
    duplicate_models = set(model_names).intersection(set(models_with_explicit_hydrogens))
    if duplicate_models:
        raise ValueError(f"Models appear in both lists: {duplicate_models}. "
                         "A model should be in either 'model_names' or 'models_with_explicit_hydrogens', not both.")
    
    # Determine if we need to process hydrogens
    need_hydrogens = len(models_with_explicit_hydrogens) > 0
    
    # Check if we have any models to evaluate
    if not model_names and not models_with_explicit_hydrogens:
        raise ValueError("No models specified for evaluation in the configuration.")
    
    results_folder = Path(config['results_folder'])
    
    # create results folder
    results_folder.mkdir(parents=True, exist_ok=True)
    
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
                add_explicit_hydrogens=need_hydrogens
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
            results = cross_validate_models(
                models=model_names,
                models_with_explicit_hydrogens=models_with_explicit_hydrogens,
                dataset=dataset,
                n_repeats=5,
                n_folds=5
            )
            
            results_df = reshape_results_to_dataframe(results)
            
            output_file = results_folder / f"{dataset_name.split('/')[-1]}_results.csv"
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error evaluating models on dataset {dataset_name}: {e}")

    logging.info("Polaris evaluation completed.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Polaris evaluation with configuration from YAML file.')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to the YAML configuration file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)
    
    main(config_path) 