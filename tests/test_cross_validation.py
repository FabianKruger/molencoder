import pytest
import numpy as np
from unittest.mock import patch
from molencoder.evaluation.cross_validation import cross_validate_models
from molencoder.utils.dummy_data import dummy_smiles_dataset


@pytest.fixture
def dummy_dataset():
    """Fixture that provides a dummy dataset for testing."""
    return dummy_smiles_dataset()


@pytest.fixture
def mock_train_and_predict():
    """Fixture that provides a mock for the train_and_predict function."""
    with patch('molencoder.evaluation.cross_validation.train_and_predict') as mock:
        # Generate random predictions and labels between 0 and 1
        mock.return_value = {
            'predictions': np.random.uniform(0, 1, 20).tolist(),
            'labels': np.random.uniform(0, 1, 20).tolist()
        }
        yield mock


def test_cross_validate_models(dummy_dataset, mock_train_and_predict):
    """Test the cross_validate_models function with dummy data and models."""
    # Define model list
    models = ["dummy1", "dummy2"]
    
    # Use 5 repeats and 5 folds as specified
    n_repeats = 5
    n_folds = 5
    
    # Call the function
    results = cross_validate_models(
        models=models,
        dataset=dummy_dataset["train"],  # Use train split from dummy dataset
        n_repeats=n_repeats,
        n_folds=n_folds,
        random_state=42
    )
    
    # Check that results have the expected structure
    assert isinstance(results, dict)
    assert set(results.keys()) == set(models)
    
    # Check that each model has the expected number of results
    # Each model should have n_repeats * n_folds results
    expected_results_count = n_repeats * n_folds
    for model in models:
        assert len(results[model]) == expected_results_count
        
        # Check that each result is a dictionary with metrics
        for result in results[model]:
            assert isinstance(result, dict)
            # Check for the actual metrics that are returned
            assert "mse" in result
            assert "mae" in result
            assert "r2" in result
            assert "rho" in result 