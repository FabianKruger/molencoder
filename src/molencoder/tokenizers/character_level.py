from tokenizers import (
    decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer, Regex
)
from pathlib import Path
import logging
import string

def create_character_level_tokenizer(save_location: Path, name: str = "character-level") -> None:
    """Creates a character-level tokenizer and saves it to the specified location."""
    logging.info("Initializing character-level tokenizer...")

    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

    # Define character sets
    uppercase_letters = list(string.ascii_uppercase)  # A-Z
    lowercase_letters = list(string.ascii_lowercase)  # a-z
    digits = list(string.digits)  # 0-9
    special_chars = list("=#:+-[]()/\\@.%")  # Explicitly listed special characters

    all_chars = uppercase_letters + lowercase_letters + digits + special_chars

    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Strip()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(r"A-Za-z0-9=#:+\-\[\]()/\\@.%"),  
        behavior="isolated",
        invert=False
    )
    trainer = trainers.WordLevelTrainer(special_tokens=special_tokens)

    # Train the tokenizer on a dummy set and add all tokens manually
    tokenizer.train_from_iterator(["CC"], trainer)
    tokenizer.add_tokens(all_chars)

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single= "[CLS]:0 $A:0 [SEP]:0",
        pair= "[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
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