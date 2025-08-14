from typing import List
import numpy as np
from datasets import Dataset


class LabelScaler:
    """
    A class for scaling labels and predictions using robust scaling (median and IQR).
    """
    
    def __init__(self, dataset: Dataset):
        """
        Initialize the LabelScaler class and calculate scaling parameters from the dataset.
        
        Args:
            dataset: A Dataset from the datasets library with a 'labels' column
        """
        labels = np.array(dataset['labels'])
        self.median = np.median(labels)
        q1 = np.percentile(labels, 25)
        q3 = np.percentile(labels, 75)
        self.iqr = q3 - q1
        if self.iqr == 0:
            self.iqr = 1  # Avoid division by zero
    
    def scale_labels(self, dataset: Dataset) -> Dataset:
        """
        Scale the 'labels' column of the dataset using the calculated median and IQR.
        
        Args:
            dataset: A Dataset from the datasets library
            
        Returns:
            The dataset with scaled labels
        """
        labels = np.array(dataset['labels'])
        scaled_labels = (labels - self.median) / self.iqr
        
        new_dataset = dataset.remove_columns(['labels'])
        new_dataset = new_dataset.add_column('labels', scaled_labels.tolist())      
        return new_dataset
    
    def scale_predictions(self, predictions: List[float]) -> List[float]:
        """
        Apply the inverse transformation to rescale predictions back to the original scale.
        
        Args:
            predictions: A list of predictions
            
        Returns:
            A list of rescaled predictions
        """
        predictions = np.array(predictions)
        rescaled_predictions = predictions * self.iqr + self.median
        return rescaled_predictions.tolist() 