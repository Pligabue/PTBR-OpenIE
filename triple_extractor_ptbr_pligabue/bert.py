from transformers import AutoTokenizer, TFAutoModel, TFBertModel

from .constants import BERT_MODEL_NAME


class Bert:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.encoder: TFBertModel = TFAutoModel.from_pretrained(BERT_MODEL_NAME).bert


bert = Bert()
