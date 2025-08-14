"""
Module for generating dummy datasets for testing.
"""
import numpy as np
from datasets import Dataset, DatasetDict


def dummy_smiles_dataset():
    """
    Fixture that provides a dummy SMILES dataset for testing.
    
    Returns:
        DatasetDict with train, validation, and test splits
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    def generate_dummy_smiles(num_examples):
        base_smiles = ["C", "CC", "CCC", "CCO", "CC(=O)O", "c1ccccc1", "C1CCCCC1"]
        return [np.random.choice(base_smiles) for _ in range(num_examples)]
    
    def generate_dummy_labels(num_examples):
        return np.random.uniform(0, 1, num_examples).tolist()
    
    train_smiles = generate_dummy_smiles(100)
    train_labels = generate_dummy_labels(100)
    
    val_smiles = generate_dummy_smiles(50)
    val_labels = generate_dummy_labels(50)
    
    test_smiles = generate_dummy_smiles(50)
    test_labels = generate_dummy_labels(50)
    
    train_dataset = Dataset.from_dict({
        "smiles": train_smiles,
        "labels": train_labels
    })
    
    validation_dataset = Dataset.from_dict({
        "smiles": val_smiles,
        "labels": val_labels
    })
    
    test_dataset = Dataset.from_dict({
        "smiles": test_smiles,
        "labels": test_labels
    })
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })
    
    return dataset_dict 