from pathlib import Path
import logging
import sqlite3
from datasets import Dataset, DatasetDict
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from tqdm import tqdm
import multiprocessing

# Setup RDKit and logging
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # Disable RDKit warnings
logging.basicConfig(level=logging.INFO)

def clean_smiles(batch):
    outputs = []
    for smi in batch["smiles"]:
        if smi is not None:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            if mol is not None:
                try:
                    rdMolStandardize.IsotopeParentInPlace(mol)  # removes all isotope labels
                    rdMolStandardize.CleanupInPlace(mol)        # cleaning steps: removing H's, sanitizing, etc.
                    rdMolStandardize.RemoveFragmentsInPlace(mol)  # removes common salts and solvents
                    canonical = Chem.MolToSmiles(mol)
                    if len(canonical) <= 500:
                        outputs.append(canonical)
                except Exception as e:
                    logging.error(f"Error processing SMILES '{smi}': {e}")
    return {"smiles": outputs}

def randomize_smiles(batch):
    outputs = []
    for smi in batch["smiles"]:
        if smi is not None:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol, doRandom=True)
                    outputs.append(smiles)
                except Exception as e:
                    logging.error(f"Error processing SMILES '{smi}': {e}")
    return {"smiles": outputs}

if __name__ == "__main__":
    chembl_db_file = Path(".../chembl_35.db")
    chembl_smiles_set = set()

    # Load ChEMBL SMILES from the database
    query = """
        SELECT DISTINCT cs.canonical_smiles
        FROM compound_structures cs
        WHERE cs.canonical_smiles IS NOT NULL;
    """
    try:
        with sqlite3.connect(chembl_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            chembl_smiles_set = set(row[0].strip() for row in rows if row[0] and len(row[0].strip()) > 0)
        logging.info(f"Loaded {len(chembl_smiles_set)} unique canonical SMILES from ChEMBL.")
    except Exception as e:
        logging.error(f"Failed to load ChEMBL data: {e}")

    # Convert the ChEMBL SMILES set to a Hugging Face dataset
    smiles_list = list(chembl_smiles_set)
    chembl_smiles_set = None  # free memory
    logging.info("Created smiles list")
    dataset = Dataset.from_dict({"smiles": smiles_list})
    logging.info(f"Initial dataset size: {len(dataset)}")
    smiles_list = None  # free memory

    # Clean the dataset
    num_proc = multiprocessing.cpu_count()
    dataset = dataset.map(clean_smiles, batched=True, num_proc=num_proc)
    logging.info(f"Amount of SMILES after cleaning: {len(dataset)}")
    
    # Deduplicate in case cleaning produced duplicates
    unique_smiles = list(set(dataset["smiles"]))
    logging.info(f"Unique SMILES after cleaning: {len(unique_smiles)}")
    dataset = Dataset.from_dict({"smiles": unique_smiles})
    unique_smiles = None  # free memory
    logging.info(f"Final dataset size: {len(dataset)}")

    # now we can randomize the SMILES
    dataset = dataset.map(randomize_smiles, batched=True, num_proc=num_proc)

    # Create a train-test split
    dataset_dict = dataset.train_test_split(shuffle=True, test_size=50000, seed=42)
    logging.info("Created train-test split.")

    # Save the split cleaned ChEMBL dataset
    dataset_dict.push_to_hub("chembl-2025-randomized-smiles-cleaned")

    dataset_dict = DatasetDict(
        train=dataset_dict["train"].select(range(len(dataset_dict["train"]) // 2)),
        test=dataset_dict["test"]
    )
    dataset_dict.push_to_hub("half-of-chembl-2025-randomized-smiles-cleaned")
    logging.info(f"Saved half_of_chembl with {len(dataset_dict['train'])} SMILES.")
