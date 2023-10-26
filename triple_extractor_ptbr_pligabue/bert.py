from transformers import AutoTokenizer, TFAutoModel, TFBertModel


class Bert:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        self.encoder: TFBertModel = TFAutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased").bert


bert = Bert()
