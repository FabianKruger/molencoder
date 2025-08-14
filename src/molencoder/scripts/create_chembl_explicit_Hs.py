from datasets import load_dataset
from rdkit import Chem
from rdkit import RDLogger
import logging
import multiprocessing

# Setup RDKit and logging
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # Disable RDKit warnings
logging.basicConfig(level=logging.INFO)

def add_explicit_hs(batch):
    outputs = []
    for smi in batch["smiles"]:
        if smi is not None:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    # Add explicit hydrogens
                    mol = Chem.AddHs(mol)
                    # Convert back to SMILES with explicit hydrogens
                    explicit_smiles = Chem.MolToSmiles(mol, doRandom=True)
                    if len(explicit_smiles) <= 500:
                        outputs.append(explicit_smiles)
                except Exception as e:
                    logging.error(f"Error processing SMILES '{smi}': {e}")
    return {"smiles": outputs}

if __name__ == "__main__":
    # Load the dataset from Hugging Face
    dataset = load_dataset("fabikru/chembl-2025-randomized-smiles-cleaned")
    initial_train_size = len(dataset["train"])
    initial_test_size = len(dataset["test"])
    logging.info(f"Initial dataset sizes - Train: {initial_train_size}, Test: {initial_test_size}")

    # Process the dataset to add explicit hydrogens
    num_proc = multiprocessing.cpu_count()
    processed_dataset = dataset.map(add_explicit_hs, batched=True, num_proc=num_proc)
    
    # Track sizes after processing
    after_processing_train_size = len(processed_dataset["train"])
    after_processing_test_size = len(processed_dataset["test"])
    logging.info(f"After processing sizes - Train: {after_processing_train_size}, Test: {after_processing_test_size}")
    logging.info(f"Lost {initial_train_size - after_processing_train_size} training examples ({((initial_train_size - after_processing_train_size) / initial_train_size * 100):.2f}%)")
    logging.info(f"Lost {initial_test_size - after_processing_test_size} test examples ({((initial_test_size - after_processing_test_size) / initial_test_size * 100):.2f}%)")

    # Save the processed dataset to Hugging Face
    processed_dataset.push_to_hub("chembl-2025-randomized-smiles-cleaned-explicit-hs")
    logging.info("Saved processed dataset to Hugging Face")
