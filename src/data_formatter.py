from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .predicate_extraction.types import SentenceId, SentenceInput, PredicateMask
from .argument_prediction.types import SubjectMask, ObjectMask

from typing import Union


class DataFormatter():
    def __init__(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> None:
        self.tokenizer = tokenizer

    def build_annotation(self, sentence_id: SentenceId, tokens: SentenceInput, pred_masks: PredicateMask,
                         subj_mask: SubjectMask, obj_mask: ObjectMask):
        pred_tokens = [t for t, mask_value in zip(tokens, pred_masks) if mask_value]
        subj_tokens = [t for t, mask_value in zip(tokens, subj_mask) if mask_value]
        obj_tokens = [t for t, mask_value in zip(tokens, obj_mask) if mask_value]
        sentence, pred, subj, obj = self.tokenizer.batch_decode(
            [tokens, pred_tokens, subj_tokens, obj_tokens],
            skip_special_tokens=True
        )
        annotation = f"Sentence {sentence_id}: {sentence}\n"
        annotation += f"Triple: <\033[92m{subj}\033[00m, \033[97m{pred}\033[00m, \033[91m{obj}\033[00m>"
        return annotation
