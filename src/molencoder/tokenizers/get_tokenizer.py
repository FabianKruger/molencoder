from typing import Literal, Optional
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from molencoder.tokenizers.character_level import create_character_level_tokenizer
from molencoder.tokenizers.schwaller import create_schwaller_tokenizer


def get_tokenizer(name: Literal["character-level", "schwaller"], folder: Path, dataset_name: Optional[Path] = None) -> PreTrainedTokenizerFast:
        
        if name == "character-level":
            tokenizer_file = folder / "character-level.json"
            if not tokenizer_file.exists():
                create_character_level_tokenizer(save_location=folder, name="character-level")
        elif name == "schwaller":
            tokenizer_file = folder / "schwaller.json"
            if not tokenizer_file.exists():
                create_schwaller_tokenizer(save_location=folder, dataset_path=dataset_name, name="schwaller", vocab_size=1024)
        else:
            raise NotImplementedError 
        pre_trained_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file),
                        unk_token="[UNK]",
                        pad_token="[PAD]",
                        cls_token="[CLS]",
                        sep_token="[SEP]",
                        mask_token="[MASK]")        
        return pre_trained_tokenizer