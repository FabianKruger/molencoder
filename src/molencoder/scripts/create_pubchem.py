from pathlib import Path
import logging
import sqlite3
from datasets import Dataset, DatasetDict
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from tqdm import tqdm

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
                    rdMolStandardize.IsotopeParentInPlace(mol) # removes all isotope labels
                    rdMolStandardize.CleanupInPlace(mol) # removing H's, sanitizing, disconnecting metals, normalizing, reioning, assigning stereochemistry
                    rdMolStandardize.RemoveFragmentsInPlace(mol) # removes common salts and solvents
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
                    if len(smiles) <= 500:
                        outputs.append(smiles)
                except Exception as e:
                    logging.error(f"Error processing SMILES '{smi}': {e}")
    return {"smiles": outputs}

if __name__ == "__main__":
    pubchem_file = Path(".../pubchem/CID-SMILES.txt")  
    chembl_db_file = Path(".../chembl_35_sqlite/chembl_35.db")
    pubchem_smiles_set = set()
    pubchem_dataset = Path(".../pubchem_smiles_random")

    # load pubchem smiles
    with open(pubchem_file, 'r', encoding='ascii') as file:
        for idx, line in enumerate(tqdm(file, desc="Loading PubChem SMILES")):
            parts = line.strip().split()
            if len(parts) < 2:
                logging.error(f"Skipping malformed line {idx}: '{line.strip()}'")
                continue
            pubchem_smiles_set.add(parts[1])
    logging.info(f"Loaded {len(pubchem_smiles_set)} unique SMILES from PubChem.")

    # load chembl smiles
    chembl_smiles_set = set()
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

    # combine the datasets
    pubchem_size = len(pubchem_smiles_set)
    chembl_size = len(chembl_smiles_set)
    pubchem_smiles_set.update(chembl_smiles_set)  # Union in-place
    combined_size = len(pubchem_smiles_set)
    overlap = pubchem_size + chembl_size - combined_size
    logging.info(f"Combined unique SMILES total: {combined_size}")
    logging.info(f"Overlap between PubChem and ChEMBL SMILES: {overlap}")

    # convert to hf dataset
    smiles_list = list(pubchem_smiles_set)
    pubchem_smiles_set = None # free memory
    logging.info("Created smiles list")     
    dataset = Dataset.from_dict({"smiles": smiles_list})
    logging.info(f"Initial dataset size: {len(dataset)}")
    smiles_list = None # free memory

    # clean the dataset
    num_proc = 16
    dataset = dataset.map(clean_smiles, batched=True, num_proc=num_proc)
    logging.info(f"Amount of SMILES after cleaning: {len(dataset)}")
    # deduplicate the dataset in case new duplicates were created during cleaning
    # needs workaround because dataset is too big for arrows internal dedpulication
    unique_smiles = list(set(dataset["smiles"]))
    logging.info(f"Unique SMILES after cleaning: {len(unique_smiles)}")
    dataset = Dataset.from_dict({"smiles": unique_smiles})
    unique_smiles = None # free memory
    dataset = dataset.map(randomize_smiles, batched=True, num_proc=num_proc)
    logging.info(f"Final dataset size: {len(dataset)}")
    dataset = dataset.train_test_split(shuffle= True, test_size=50000, seed=42)

    # save dataset 
    dataset.save_to_disk(str(pubchem_dataset))
