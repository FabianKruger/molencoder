import polaris as po
from datasets import Dataset
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import logging

# Setup RDKit logging
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # Disable RDKit warnings


def add_hydrogens(dataset: Dataset) -> Dataset:
    """
    Adds a new column 'smiles_with_hydrogens' containing SMILES strings with explicit hydrogens.
    
    Args:
        dataset (Dataset): Hugging Face dataset with 'smiles' column
        
    Returns:
        Dataset: Dataset with added 'smiles_with_hydrogens' column
        
    Raises:
        ValueError: If any SMILES cannot be converted to explicit hydrogens
    """
    def process_smiles_batch(examples):
        smiles_with_h = []
        
        for i, smiles in enumerate(examples["smiles"]):
            if smiles is None:
                raise ValueError(f"SMILES string is None at index {i}")
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Failed to parse SMILES: '{smiles}' at index {i}")
                    
                # Add explicit hydrogens
                mol = Chem.AddHs(mol)
                
                # Convert back to SMILES with explicit hydrogens
                explicit_smiles = Chem.MolToSmiles(mol, doRandom=True)
                
                if explicit_smiles is None or explicit_smiles == "":
                    raise ValueError(f"Failed to generate explicit hydrogens for SMILES: '{smiles}' at index {i}")
                
                smiles_with_h.append(explicit_smiles)
                    
            except Exception as e:
                if isinstance(e, ValueError):
                    raise e
                else:
                    raise ValueError(f"Error processing SMILES '{smiles}' at index {i}: {e}") from e
        
        examples["smiles_with_hydrogens"] = smiles_with_h
        return examples
    
    logging.info("Adding explicit hydrogens to dataset...")
    dataset_with_h = dataset.map(process_smiles_batch, batched=True)
    
    logging.info(f"✅ Successfully added explicit hydrogens to all {len(dataset_with_h)} molecules")
    return dataset_with_h


def preprocess_polaris_benchmark(benchmark_name, add_explicit_hydrogens=False) -> Dataset:
    """
    Loads a Polaris benchmark dataset, combines train and test data,
    and converts it to a Hugging Face dataset with standardized column names.
    
    Args:
        benchmark_name (str): Name of the Polaris benchmark to load
        add_explicit_hydrogens (bool, optional): Whether to add explicit hydrogens column. Defaults to False.
        
    Returns:
        Dataset: Hugging Face dataset with 'smiles', 'labels', and optionally 'smiles_with_hydrogens' columns
        
    Raises:
        RuntimeError: If the user hasn't logged in to Polaris first or if the dataset name is incorrect
        ValueError: If adding explicit hydrogens fails
    """
    # Load the benchmark
    try:
        benchmark = po.load_benchmark(benchmark_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load benchmark '{benchmark_name}'. "
            "Please make sure you have logged in to Polaris first using 'polaris login' "
            "and verify that the dataset name is correct. You can list available datasets using 'polaris benchmark list'."
        ) from e
    
    train, _ = benchmark.get_train_test_split()
    train_df = train.as_dataframe()
    dataset = Dataset.from_pandas(train_df)
    
    # Get current column names
    current_cols = list(dataset.column_names)
    
    # Only rename columns if they don't already have the correct names
    if "smiles" not in current_cols or "labels" not in current_cols:
        smiles_col, label_col = determine_columns(dataset)
        if "smiles" not in current_cols:
            dataset = dataset.rename_column(smiles_col, "smiles")
        if "labels" not in current_cols:
            dataset = dataset.rename_column(label_col, "labels")
    
    # Optionally add explicit hydrogens as a new column
    if add_explicit_hydrogens:
        dataset = add_hydrogens(dataset)
    
    return dataset


def determine_columns(dataset):
    """
    Determines which column contains the SMILES strings and which contains the labels.
    Assumes the dataset has exactly two columns and that the SMILES column is the only non-numeric one.
    
    Args:
        dataset (Dataset): Hugging Face dataset
        
    Returns:
        tuple: (smiles_column_name, label_column_name)
    """
    row = dataset[0]
    cols = list(row.keys())
    if len(cols) != 2:
        raise ValueError("Dataset must have exactly two columns.")
    
    smiles_col, label_col = None, None
    for col in cols:
        if isinstance(row[col], str):
            smiles_col = col
        else:
            label_col = col
    if smiles_col is None or label_col is None:
        raise ValueError("Could not determine columns based on non-numeric criterion.")
    
    return smiles_col, label_col 