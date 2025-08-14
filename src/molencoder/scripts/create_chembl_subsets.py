from datasets import load_dataset, DatasetDict
import logging

logging.basicConfig(level=logging.INFO)

def create_smaller_subset(dataset_name, new_name, factor=2):
    """Create a smaller subset by taking 1/factor of the training data."""
    logging.info(f"Loading {dataset_name}...")
    dataset = load_dataset(f"fabikru/{dataset_name}")
    
    # Calculate new size
    new_train_size = len(dataset["train"]) // factor
    logging.info(f"Creating {new_name} with {new_train_size} training examples")
    
    # Create new dataset with smaller training set
    new_dataset = dataset["train"].select(range(new_train_size))
    new_dataset_dict = DatasetDict({
        "train": new_dataset,
        "test": dataset["test"]
    })
    
    logging.info(new_dataset_dict)
    # Push to hub
    new_dataset_dict.push_to_hub(f"fabikru/{new_name}")
    logging.info(f"Successfully uploaded {new_name}")

if __name__ == "__main__":
    create_smaller_subset(
        "half-of-chembl-2025-randomized-smiles-cleaned",
        "quarter-of-chembl-2025-randomized-smiles-cleaned"
    )
    
    create_smaller_subset(
        "quarter-of-chembl-2025-randomized-smiles-cleaned",
        "eighth-of-chembl-2025-randomized-smiles-cleaned"
    )
