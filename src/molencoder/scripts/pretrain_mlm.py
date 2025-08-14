import argparse
import json
import logging
import yaml
import numpy as np
from pathlib import Path
from transformers import (ModernBertConfig,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          ModernBertForMaskedLM,
                          Trainer,
                          DataCollatorWithPadding,
                          EarlyStoppingCallback)
from datasets import DatasetDict, load_dataset
from huggingface_hub import login
from molencoder.tokenizers.get_tokenizer import get_tokenizer
from molencoder.utils.callbacks import CSVLoggerCallback
import torch
from transformers import logging as hf_logging
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
hf_logging.set_verbosity_info()

# Increase torch._dynamo recompilation limit to handle variable sequence lengths
import torch._dynamo
torch._dynamo.config.cache_size_limit = 32  # Default is 8


def run_trial(config: dict):
    # Extract parameters from config.
    dataset_name = config["dataset_name"]
    tokenizer_name = config["tokenizer_name"]
    debug = config["debug"]
    tokenizer_path = Path(config["tokenizer_folder_path"])

    # Extract path from the config.
    results_path = Path(config["result_folder_path"])

    # Automatically assign the output_dir for TrainingArguments.
    config["training_arguments"]["output_dir"] = str(results_path / "trainer")
    config["training_arguments"]["logging_dir"] = str(results_path / "training")
   
    training_args = TrainingArguments(**config["training_arguments"])
    
    # Load dataset.
    dataset = load_dataset(dataset_name)
    logging.info("Dataset is loaded.")
    
    if debug:
        # For debugging, use smaller dataset sizes
        dataset = DatasetDict({
            "train": dataset["train"].shuffle().select(range(5000)),
            "test": dataset["test"].shuffle().select(range(500)),
        })

    # Load or create the tokenizer.
    tokenizer = get_tokenizer(name=tokenizer_name, folder=tokenizer_path, dataset_name=dataset_name)

    # Add to the  the model configuration and initialize it
    config["modern_bert_config"]["pad_token_id"] = tokenizer.vocab.get("[PAD]")
    config["modern_bert_config"]["cls_token_id"] = tokenizer.vocab.get("[CLS]")
    config["modern_bert_config"]["sep_token_id"] = tokenizer.vocab.get("[SEP]")
    config["modern_bert_config"]["vocab_size"] = len(tokenizer)   # Set the vocab size dynamically.
    model_config = ModernBertConfig(**config["modern_bert_config"])

    # Tokenize the dataset.
    # Original implementation (without fixed lengths)
    # def tokenize(element):
    #     outputs = tokenizer(
    #         element["smiles"],
    #         truncation=False,
    #         max_length=model_config.max_position_embeddings
    #     )
    #     return {"input_ids": outputs["input_ids"]}
    
    # Modified implementation with fixed-length sequences
    def tokenize(element):
        outputs = tokenizer(
            element["smiles"],
            truncation=True,  # Enable truncation
            padding="max_length",  # Add padding
            max_length=model_config.max_position_embeddings - 2  # Account for special tokens
        )
        return {"input_ids": outputs["input_ids"]}
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["smiles"], num_proc=7)

    # Define a data collator for training (with random masking).
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=config["masking_probability"])

    # Set the vocab size dynamically.
    model = ModernBertForMaskedLM(config=model_config)
    model_size = sum(t.numel() for t in model.parameters())/1000**2
    logging.info(f"Model size: {model_size:.1f} M parameters")
    
    def preprocess_logits_for_metrics(logits, labels):
        # Compute predicted token IDs (batch_size x seq_length)
        pred_ids = torch.argmax(logits, dim=-1)
        # Create a mask to ignore tokens with label -100
        mask = labels != -100
        # Count correct predictions in the batch (as a torch tensor)
        batch_correct = (pred_ids == labels)[mask].sum()
        # Count total valid tokens in the batch (as a torch tensor)
        batch_total = mask.sum()
        # Return the counts as torch tensors (do not call .item())
        return batch_correct, batch_total

    
    def compute_metrics(eval_pred):
        preprocessed_metrics, _ = eval_pred # very annoying that hugging face trainer gives all the labels anyway again here
        batch_corrects, batch_totals = preprocessed_metrics
        # Use torch operations directly
        total_correct = batch_corrects.sum().item()
        total_tokens = batch_totals.sum().item()
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        return {"accuracy": accuracy}


    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)
    csv_logger_callback = CSVLoggerCallback(output_dir=results_path)

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=tokenized_dataset["train"],
                      eval_dataset=tokenized_dataset["test"],
                      compute_metrics=compute_metrics,
                      callbacks=[early_stopping_callback, csv_logger_callback],
                      preprocess_logits_for_metrics=preprocess_logits_for_metrics)
    
    trainer.train()
    eval_results = trainer.evaluate()
    logging.info("Final evaluation results on validation set are:\n" + json.dumps(eval_results, indent=2))

    # Save the model and trainer state to a subfolder "model" in the result folder.
    model_save_dir = results_path / "model"
    model_save_dir.mkdir(exist_ok=True, parents=True)
    trainer.save_model(str(model_save_dir))
    trainer.state.save_to_json(str(model_save_dir / "trainer_state.json"))
    logging.info(f"Saved model and trainer state to {model_save_dir}")

    # Write test evaluation results to disk as YAML.
    eval_file = results_path / "final_evaluation_results.yaml"
    with eval_file.open("w") as f:
        yaml.dump(eval_results, f)
    logging.info(f"Saved test evaluation results to {eval_file}")

    # Determine dataset size label for naming
    def get_dataset_size_label(dataset_name):
        if "eighth-of-chembl" in dataset_name:
            return "smallest"
        elif "quarter-of-chembl" in dataset_name:
            return "smaller"
        elif "half-of-chembl" in dataset_name:
            return "small"
        elif "pubchem-1M-randomized-smiles-cleaned" in dataset_name:
            return "pubchem_1M"
        elif "chembl-1M-randomized-smiles-cleaned" in dataset_name:
            return "chembl_1M"
        else:
            # Fallback for other datasets
            return "unknown"
    
    dataset_size_label = get_dataset_size_label(dataset_name)
    model_name = f"model_15M_{dataset_size_label}_ds_masking_{config['masking_probability']}_predicted_hparams"

    # Push the best model and tokenizer to the Hugging Face Hub
    # Note: trainer.model contains the best model due to load_best_model_at_end=True
    login(token=config["huggingface_token"])
    logging.info(f"Pushing best model with name: {model_name}")
    
    trainer.model.push_to_hub(
        model_name,
        commit_message=f"Upload best {dataset_size_label} dataset model (masking={config['masking_probability']})"
    )
    
    tokenizer.push_to_hub(
        model_name,
        commit_message=f"Upload tokenizer for {dataset_size_label} dataset model"
    )
    
    logging.info("Pushed best model and tokenizer to the Hugging Face Hub")

def main():
    parser = argparse.ArgumentParser(description="Run trial for masked LM training on the ChEMBL dataset")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load the YAML configuration from the provided path.
    with args.config.open("r") as file:
        config = yaml.safe_load(file)

    # Save a copy of the complete configuration in the result folder.
    result_folder = Path(config["result_folder_path"])
    result_folder.mkdir(exist_ok=True, parents=True)
    experiment_config_path = result_folder / "experiment_config.yaml"
    with experiment_config_path.open("w") as file:
        yaml.dump(config, file)
    logging.info(f"Saved experiment configuration to {experiment_config_path}")

    # Run the trial using the complete config.
    run_trial(config=config)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()