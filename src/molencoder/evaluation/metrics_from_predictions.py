import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from typing import Dict, List


def metrics_from_predictions(
    predictions_dict: Dict[str, List[float]]
) -> Dict[str, float]:
    """
    Calculate regression metrics from predictions and labels.
    
    Args:
        predictions_dict: Dictionary containing 'predictions' and 'labels' lists
        
    Returns:
        Dictionary containing the calculated metrics
    """
    predictions = np.array(predictions_dict['predictions'])
    labels = np.array(predictions_dict['labels'])
    
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    rho, _ = spearmanr(labels, predictions)
    
    return {
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'rho': rho
    }
