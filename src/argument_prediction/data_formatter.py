from transformers import AutoTokenizer, TFAutoModel, TFBertModel


class DataFormatter():
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        self.bert: TFBertModel = TFAutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
