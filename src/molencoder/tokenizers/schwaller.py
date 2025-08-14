from tokenizers import (
    decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer, Regex
)
from pathlib import Path
import logging
from datasets import load_dataset

def create_schwaller_tokenizer(
    save_location: Path,
    dataset_name: Path, 
    name: str = "schwaller", 
    vocab_size: int = 1024
) -> None:
    """Creates a tokenizer based on Schwaller et al. and saves it as a JSON file."""
    
    logging.info("Loading dataset...")
    dataset = load_dataset(dataset_name)

    def get_training_corpus():
        """Yields SMILES strings from the dataset in chunks of 1000."""
        for split in ["train", "test"]:
            ds = dataset[split] 
            for i in range(0, len(ds), 1000):
                yield ds[i: i + 1000]["smiles"]

    logging.info("Initializing tokenizer based on Schwaller et al. ...")
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Strip()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(r"(\[[^]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|%\d+|\d)"),  
        behavior="isolated",
        invert=False
    )
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    logging.info("Starting tokenizer training...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer)

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )
    tokenizer.decoder = decoders.Strip()
    
    tokenizer_path = save_location / f"{name}.json"
    try:
        tokenizer.save(str(tokenizer_path))
        logging.info(f"Tokenizer saved successfully to {tokenizer_path}")
    except Exception as e:
        logging.error(f"Failed to save tokenizer: {e}")

    logging.info("Tokenizer creation process completed.")