import re
import pandas as pd
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from nltk.tokenize import sent_tokenize

from typing import Union

from .predicate_extraction.types import SentenceId, SentenceInput, PredicateMask
from .argument_prediction.types import ArgPredOutput, ArgPredOutputs, SubjectMask, ObjectMask


class DataFormatter():
    COLUMNS = ['confidence', 'subject', 'relation', 'object', 'subject_id', 'object_id']

    def __init__(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> None:
        self.tokenizer = tokenizer

    def build_annotation(self, sentence_id: SentenceId, tokens: SentenceInput, pred_mask: PredicateMask,
                         subj_mask: SubjectMask, obj_mask: ObjectMask):
        pred_tokens = [t for t, mask_value in zip(tokens, pred_mask) if mask_value]
        subj_tokens = [t for t, mask_value in zip(tokens, subj_mask) if mask_value]
        obj_tokens = [t for t, mask_value in zip(tokens, obj_mask) if mask_value]
        sentence, pred, subj, obj = self.tokenizer.batch_decode(
            [tokens, pred_tokens, subj_tokens, obj_tokens],
            skip_special_tokens=True
        )
        annotation = f"Sentence {sentence_id}: {sentence}\n"
        annotation += f"Triple: <\033[92m{subj}\033[00m, \033[97m{pred}\033[00m, \033[91m{obj}\033[00m>"
        return annotation

    def build_id(self, id_prefix: str, sentence_id: SentenceId, label: str,
                 mask: Union[SubjectMask, PredicateMask, ObjectMask]):
        sentence_chuck = f"S{sentence_id}"
        mask_chunk = "-".join([f"{i}" for i, v in enumerate(mask) if v])
        label_chunk = re.sub(r"\s", '-', label)
        return f"{id_prefix}-{sentence_chuck}-{mask_chunk}-{label_chunk}"

    def build_element(self, column: str, subj_label: str, pred_label: str, obj_label: str, subj_id: str, obj_id: str):
        if column == "confidence":
            return 1.0
        if column == "subject":
            return subj_label
        if column == "relation":
            return pred_label
        if column == "object":
            return obj_label
        if column == "subject_id":
            return subj_id
        if column == "object_id":
            return obj_id
        raise Exception(f"Column <{column}> has no matched operation")

    def build_row(self, output: ArgPredOutput, id_prefix=''):
        sentence_id, tokens, pred_mask, subj_mask, obj_mask = output

        pred_tokens = [t for t, mask_value in zip(tokens, pred_mask) if mask_value]
        subj_tokens = [t for t, mask_value in zip(tokens, subj_mask) if mask_value]
        obj_tokens = [t for t, mask_value in zip(tokens, obj_mask) if mask_value]
        pred, subj, obj = self.tokenizer.batch_decode(
            [pred_tokens, subj_tokens, obj_tokens],
            skip_special_tokens=True
        )
        subj_id = self.build_id(id_prefix, sentence_id, subj, subj_mask)
        obj_id = self.build_id(id_prefix, sentence_id, subj, obj_mask)

        return [self.build_element(column, subj, pred, obj, subj_id, obj_id) for column in self.COLUMNS]

    def build_df(self, outputs: ArgPredOutputs, id_prefix='') -> pd.DataFrame:
        data = [self.build_row(output, id_prefix=id_prefix) for output in outputs]
        return pd.DataFrame(columns=self.COLUMNS, data=data)

    def doc_to_sentences(self, doc: str):
        return sent_tokenize(doc, language="portuguese")
