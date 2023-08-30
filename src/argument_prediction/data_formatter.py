import re
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, TFBertModel

from typing import Union

from ..predicate_extraction.types import SentenceInput, SentenceInputs, PredicateMasks
from ..constants import (MAX_SENTENCE_SIZE, OBJECT_PATTERN, PREDICATE_PATTERN,
                         SPECIAL_TOKEN_IDS, SUBJECT_PATTERN)

from .types import (BIO, TrainingData, FormattedSentenceOutput, FormattedTokenOutput, SentenceMap,
                    SentenceMapValue, Span)


class DataFormatter():
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        self.bert: TFBertModel = TFAutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

    #################
    # INPUT SECTION #
    #################

    def unannotate_sentence(self, sentence: str):
        return re.sub(r"<[^>]*>([^<]*)</[^>]*>", r"\1", sentence)

    def split_on(self, sentence, pattern) -> list[str]:
        split_sentence = re.split(pattern, sentence)
        trimmed_split = [chunk.strip() for chunk in split_sentence]
        unannotated_split = [self.unannotate_sentence(chunk) for chunk in trimmed_split]
        return unannotated_split

    def get_span(self, sentence, pattern) -> Span:
        split_sentence = self.split_on(sentence, pattern)
        token_sets: list = self.tokenizer(split_sentence, add_special_tokens=False)["input_ids"]  # type: ignore
        start = len(token_sets[0]) + 1  # 1 is added to account for the [CLS] token that is added later
        end = start + len(token_sets[1])
        return (start, end)

    def span_to_mask(self, span):
        start, finish = span
        return [start <= i < finish for i in range(MAX_SENTENCE_SIZE)]

    def tokenize(self, sen: Union[list[str], str]):
        if isinstance(sen, list):
            return self.tokenizer(sen, padding="max_length", max_length=MAX_SENTENCE_SIZE)["input_ids"]
        return self.tokenizer.encode(sen, padding="max_length", max_length=MAX_SENTENCE_SIZE)

    def token_to_tag(self, token, index, subject_span: Span, object_span: Span):
        if index in range(*subject_span):
            return BIO.SB if index == subject_span[0] else BIO.SI
        if index in range(*object_span):
            return BIO.OB if index == object_span[0] else BIO.OI
        if hasattr(BIO, "S") and token in SPECIAL_TOKEN_IDS:
            return BIO.S
        return BIO.O

    def add_training_sentence_data(self, key: str, annotated_sentence: str, input_tokens: SentenceInput,
                                   sentence_map: SentenceMap):
        if key not in sentence_map:
            initial_output = tf.zeros((len(input_tokens), len(BIO)))
            sentence_map[key] = {"output": initial_output, "count": 0}

        subject_span = self.get_span(annotated_sentence, SUBJECT_PATTERN)
        object_span = self.get_span(annotated_sentence, OBJECT_PATTERN)
        output_tags = [self.token_to_tag(t, i, subject_span, object_span) for i, t in enumerate(input_tokens)]
        encoded_output_tags = tf.one_hot([tag.value for tag in output_tags], len(BIO))
        sentence_map[key]["output"] += encoded_output_tags
        sentence_map[key]["count"] += 1

    def normalize_training_output(self, sentence_map: SentenceMapValue, weighted=False) -> tf.Tensor:
        if not weighted:
            clipped = tf.clip_by_value(sentence_map["output"], 0.0, 1.0)
            normalization_matrix = tf.math.reduce_sum(clipped, axis=1, keepdims=True)
            return clipped / normalization_matrix
        return sentence_map["output"] / sentence_map["count"]  # type: ignore

    def format_training_data(self, sentences: list[str], merge_repeated=True,
                             weighted_merge=False) -> TrainingData:
        unannotated_sentences = [self.unannotate_sentence(sen) for sen in sentences]
        sentence_inputs: SentenceInputs = self.tokenize(unannotated_sentences)  # type: ignore
        pred_spans = [self.get_span(sentence, PREDICATE_PATTERN) for sentence in sentences]
        predicate_masks: PredicateMasks = [self.span_to_mask(span) for span in pred_spans]

        sentence_map: SentenceMap = {}  # Keeps track of sentences that are repeated with different predicates
        keys = unannotated_sentences if merge_repeated else sentences
        for key, sentence, tokens in zip(keys, sentences, sentence_inputs):
            self.add_training_sentence_data(key, sentence, tokens, sentence_map)

        training_x = [tf.constant(sentence_inputs), tf.constant(predicate_masks)]
        training_y = [self.normalize_training_output(sentence_map[key], weighted_merge) for key in keys]
        return training_x, tf.stack(training_y)

    ##################
    # OUTPUT SECTION #
    ##################

    def format_output(self, sentence_output) -> FormattedSentenceOutput:
        sentence_result: FormattedSentenceOutput = []
        max_indexes = tf.math.top_k(sentence_output, len(BIO)).indices.numpy()
        for token_output, max_index in zip(sentence_output, max_indexes):
            result: FormattedTokenOutput = []
            for i in max_index:
                tag = BIO(i)
                score = token_output[i]
                result.append((tag, score))
            sentence_result.append(result)
        return sentence_result

    def format_outputs(self, outputs):
        return [self.format_output(output) for output in outputs]
