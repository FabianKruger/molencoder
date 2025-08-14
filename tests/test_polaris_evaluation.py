import pytest
import pandas as pd
from pathlib import Path
import os
import yaml
import argparse
from unittest.mock import MagicMock, patch

from molencoder.scripts.polaris_evaluation import (
    main, 
    reshape_results_to_dataframe, 
    load_config,
    parse_args
)


@pytest.fixture
def config_data(tmp_path):
    """Fixture that provides sample configuration data."""
    return {
        'dataset_names': ["solubility", "logp"],
        'model_names': ["model1", "model2"],
        'results_folder': str(tmp_path / "test_results")
    }


@pytest.fixture
def config_file(tmp_path, config_data):
    """Fixture that creates a temporary config file."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def mock_dataset():
    """Fixture that provides a mock dataset."""
    dataset = MagicMock()
    dataset.features = {"smiles": None, "labels": None}
    dataset.__len__.return_value = 100
    return dataset


@pytest.fixture
def mock_results():
    """Fixture that provides mock results from cross_validate_models."""
    return {
        "model1": [
            {"rmse": 0.5, "mae": 0.4, "r2": 0.8},
            {"rmse": 0.6, "mae": 0.5, "r2": 0.7}
        ],
        "model2": [
            {"rmse": 0.7, "mae": 0.6, "r2": 0.6},
            {"rmse": 0.8, "mae": 0.7, "r2": 0.5}
        ]
    }


def test_reshape_results_to_dataframe(mock_results):
    """Test the reshape_results_to_dataframe function."""
    df = reshape_results_to_dataframe(mock_results)
    
    # Check that the DataFrame has the expected columns
    assert "model" in df.columns
    assert "fold" in df.columns
    assert "metric_name" in df.columns
    assert "value" in df.columns
    
    # Check that the DataFrame has the expected number of rows
    # Calculate the total number of metrics across all folds for all models
    expected_rows = sum(len(metrics) * len(metrics[0]) for metrics in mock_results.values())
    assert len(df) == expected_rows
    
    # Check that all models are in the DataFrame
    assert set(df["model"].unique()) == set(mock_results.keys())
    
    # Check that all metrics are in the DataFrame
    expected_metrics = set()
    for metrics_list in mock_results.values():
        for metrics in metrics_list:
            expected_metrics.update(metrics.keys())
    assert set(df["metric_name"].unique()) == expected_metrics


def test_load_config(config_file, config_data):
    """Test the load_config function."""
    loaded_config = load_config(config_file)
    assert loaded_config == config_data


def test_parse_args():
    """Test the parse_args function."""
    test_args = ["--config", "test_config.yaml"]
    with patch('sys.argv', ['script_name'] + test_args):
        args = parse_args()
        assert args.config == "test_config.yaml"


def test_main_function(monkeypatch, tmp_path, config_file, config_data, mock_dataset):
    """Test the main function with mocked dependencies."""
    # Create mock functions
    mock_preprocess = MagicMock(return_value=mock_dataset)
    # For this test, we just need cross_validate_models to return something that can be reshaped
    # We don't need the full mock_results structure
    mock_cross_validate = MagicMock(return_value={
        "model1": [{"metric": 0.5}],
        "model2": [{"metric": 0.6}]
    })
    
    # Apply monkeypatch - patch the functions in the polaris module's namespace
    monkeypatch.setattr("molencoder.scripts.polaris_evaluation.preprocess_polaris_benchmark", mock_preprocess)
    monkeypatch.setattr("molencoder.scripts.polaris_evaluation.cross_validate_models", mock_cross_validate)
    
    # Call the main function
    main(config_file)
    
    # Check that preprocess_polaris_benchmark was called for each dataset
    assert mock_preprocess.call_count == len(config_data['dataset_names'])
    for dataset_name in config_data['dataset_names']:
        mock_preprocess.assert_any_call(dataset_name, add_explicit_hydrogens=False)
    
    # Check that cross_validate_models was called for each dataset
    assert mock_cross_validate.call_count == len(config_data['dataset_names'])
    for dataset_name in config_data['dataset_names']:
        mock_cross_validate.assert_any_call(
            models=config_data['model_names'],
            models_with_explicit_hydrogens=[],
            dataset=mock_dataset,
            n_repeats=5,
            n_folds=5
        )
    
    # Check that CSV files were created for each dataset
    results_folder = Path(config_data['results_folder'])
    for dataset_name in config_data['dataset_names']:
        output_file = results_folder / f"{dataset_name}_results.csv"
        assert output_file.exists()
        
        # Check that the CSV file contains the expected data
        df = pd.read_csv(output_file)
        assert "model" in df.columns
        assert "fold" in df.columns
        assert "metric_name" in df.columns
        assert "value" in df.columns
    
    # Check that config file was copied to results folder
    config_copy_path = results_folder / 'config.yaml'
    assert config_copy_path.exists()
    with open(config_copy_path, 'r') as f:
        copied_config = yaml.safe_load(f)
    assert copied_config == config_data
