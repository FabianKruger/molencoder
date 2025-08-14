#!/usr/bin/env python

import tempfile
from typing import Dict, List
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TransformerBaseline:
    """
    Transformer baseline wrapper for molecular property prediction.
    
    This class provides a consistent interface for training transformer models
    (including the user's model and ChemBERTa models) using the same hyperparameters
    and training procedure as the existing train_and_predict function.
    """
    
    def __init__(self):
        """Initialize the transformer baseline."""
        pass
    
    def train_and_predict(
        self,
        model_name: str,
        dataset_dict: DatasetDict,
        smiles_column: str = 'smiles'
    ) -> Dict[str, List[float]]:
        """
        Train a transformer model on the provided dataset using fixed hyperparameters.
        
        This method uses the same hyperparameters and training procedure as the
        existing train_and_predict function to ensure fair comparison.
        
        Args:
            model_name: Name or path of the pretrained model
            dataset_dict: DatasetDict containing 'train', 'validation', and 'test' splits
            smiles_column: Name of the column containing SMILES strings (default: 'smiles')
            
        Returns:
            Dictionary containing the predictions and labels
        """

        required_splits = ["train", "validation", "test"]
        for split in required_splits:
            if split not in dataset_dict:
                raise ValueError(f"Dataset must contain a '{split}' split")
        
        required_columns = [smiles_column, "labels"]
        for split in required_splits:
            for column in required_columns:
                if column not in dataset_dict[split].features:
                    raise ValueError(f"Dataset '{split}' split must contain a '{column}' column")
        
        trust_remote_code = "MoLFormer" in model_name or "molformer" in model_name.lower()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        def tokenize_function(examples):
            return tokenizer(examples[smiles_column], truncation=True, max_length=500)
        dataset_dict = dataset_dict.map(tokenize_function, batched=True)
        

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            trust_remote_code=trust_remote_code,
        )
        

        with tempfile.TemporaryDirectory() as temp_dir:
            training_args = TrainingArguments(
                num_train_epochs=500,
                per_device_train_batch_size=64,
                per_device_eval_batch_size=64,
                optim="schedule_free_adamw",
                lr_scheduler_type="constant",
                learning_rate=8e-4,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-8,
                weight_decay=1e-5,
                fp16=False,
                bf16=True,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                dataloader_num_workers=8,
                dataloader_pin_memory=True,
                warmup_steps=100,
                eval_on_start=False,
                output_dir=temp_dir,
                logging_dir=temp_dir,
                tf32=True,
                save_total_limit=2,
                torch_compile=True,
                torch_compile_backend="inductor",
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                max_grad_norm=1.0,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset_dict["train"],
                eval_dataset=dataset_dict["validation"],
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(
                    early_stopping_patience=5,
                    early_stopping_threshold=0.001
                )],
            )
            trainer.train()

            combined_output = trainer.predict(dataset_dict["test"])
            predictions = combined_output.predictions.flatten().tolist()
            labels = combined_output.label_ids.flatten().tolist()
        
        return {"predictions": predictions, "labels": labels} 