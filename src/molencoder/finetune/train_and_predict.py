import tempfile
from typing import Dict, List, Tuple
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

def train_and_predict(
    model_name: str,
    dataset_dict: DatasetDict,
    smiles_column: str = 'smiles'
) -> Dict[str, List[float]]:
    """
    Train a model on the provided dataset using fixed hyperparameters.
    
    Args:
        model_name: Name or path of the pretrained model
        dataset_dict: DatasetDict containing 'train' and 'validation' splits
        smiles_column: Name of the column containing SMILES strings (default: 'smiles')
        
    Returns:
        Dictionary containing the predictions and labels
    """
    # Check if dataset has the required splits
    required_splits = ["train", "validation"]
    for split in required_splits:
        if split not in dataset_dict:
            raise ValueError(f"Dataset must contain a '{split}' split")
    required_columns = [smiles_column, "labels"]
    for split in required_splits:
        for column in required_columns:
            if column not in dataset_dict[split].features:
                raise ValueError(f"Dataset '{split}' split must contain a '{column}' column")
    
    # Tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples[smiles_column], truncation=True, max_length=500)
    dataset_dict = dataset_dict.map(tokenize_function, batched=True)
    
    # load model and data collator
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # Regression task
    )
    
    # Train the model with a temporary directory for training outputs
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

        combined_output= trainer.predict(dataset_dict["test"])
        predictions = combined_output.predictions.flatten().tolist()
        labels = combined_output.label_ids.flatten().tolist()
    
    return {"predictions": predictions, "labels": labels} 
