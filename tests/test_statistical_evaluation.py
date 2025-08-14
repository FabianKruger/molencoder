import pytest
import numpy as np
import pandas as pd
from molencoder.evaluation.statistical_evaluation import repeated_measures_anova, tukey_hsd


@pytest.fixture
def three_model_data():
    """Fixture that provides a DataFrame with three models and 5 measurements each."""
    return pd.DataFrame({
        "model1": [0.1, 0.12, 0.11, 0.13, 0.14],
        "model2": [0.15, 0.16, 0.17, 0.18, 0.19],
        "model3": [0.2, 0.21, 0.22, 0.23, 0.24]
    })


def test_repeated_measures_anova(three_model_data):
    """Test the repeated_measures_anova function with three models."""
    # Run the ANOVA
    results = repeated_measures_anova(three_model_data)
    
    # Check that the function returns a dictionary with the expected keys
    assert isinstance(results, dict)
    assert 'f_statistic' in results
    assert 'p_value' in results
    assert 'anova_table' in results
    
    # Check that the values have the expected types
    assert isinstance(results['f_statistic'], float)
    assert isinstance(results['p_value'], float)
    assert isinstance(results['anova_table'], pd.DataFrame)
    
    # Check that p_value is between 0 and 1
    assert 0 <= results['p_value'] <= 1
    
    # Check that the ANOVA table has the expected columns
    expected_columns = ['F Value', 'Num DF', 'Den DF', 'Pr > F']
    for col in expected_columns:
        assert col in results['anova_table'].columns


def test_tukey_hsd(three_model_data):
    """Test the tukey_hsd function with three models."""
    # Run the Tukey HSD test
    results = tukey_hsd(three_model_data)
    
    # Check that the function returns a dictionary with the expected keys
    assert isinstance(results, dict)
    assert 'results' in results
    assert 'summary_table' in results
    
    # Check that the values have the expected types
    assert hasattr(results['results'], 'summary')
    assert isinstance(results['summary_table'], pd.DataFrame)
    
    # Check that the summary table has the expected columns
    expected_columns = ['group1', 'group2', 'meandiff', 'p-adj', 'lower', 'upper', 'reject']
    for col in expected_columns:
        assert col in results['summary_table'].columns
    
    # Check that the number of comparisons is correct (n*(n-1)/2 where n is the number of models)
    n_models = len(three_model_data.columns)
    expected_comparisons = n_models * (n_models - 1) / 2
    assert len(results['summary_table']) == expected_comparisons
    
    # Check that the alpha parameter works
    custom_alpha = 0.01
    custom_results = tukey_hsd(three_model_data, alpha=custom_alpha)
    assert isinstance(custom_results, dict)
    assert 'results' in custom_results
    assert 'summary_table' in custom_results 