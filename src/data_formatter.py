from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from typing import Union


class DataFormatter():
    def __init__(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> None:
        self.tokenizer = tokenizer
