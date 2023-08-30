import re
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

from typing import Union

from ..constants import (MAX_SENTENCE_SIZE, OBJECT_PATTERN, PREDICATE_PATTERN,
                         SPECIAL_TOKEN_IDS, SUBJECT_PATTERN)

from .types import (BIO, SentenceMap, SentenceInput, SentenceInputs,
                    FormattedTokenOutput,FormattedSentenceOutput)


class DataFormatter():
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

    #################
    # INPUT SECTION #
    #################

    def split_on_predicate(self, sentence) -> list[str]:
        sentence = re.sub(SUBJECT_PATTERN, r"\1", sentence)
        sentence = re.sub(OBJECT_PATTERN, r"\1", sentence)
        split_sentence = re.split(PREDICATE_PATTERN, sentence)
        trimmed_split = [chunk.strip() for chunk in split_sentence]
        return trimmed_split

    def format_input(self, sentence: str) -> SentenceInput:
        return self.tokenizer.encode(sentence, padding="max_length", max_length=MAX_SENTENCE_SIZE)

    def format_inputs(self, sentences: list[str]) -> SentenceInputs:
        return [self.format_input(sentence) for sentence in sentences]

    def token_to_tag(self, token, index, predicate_start, predicate_end):
        if index == predicate_start:
            return BIO.B
        if predicate_start < index < predicate_end:
            return BIO.I
        if hasattr(BIO, "S") and token in SPECIAL_TOKEN_IDS:
            return BIO.S
        return BIO.O

    def add_training_sentence_data(self, sentence: str, sentence_map: SentenceMap, merge_repeated=True):
        split_sentence = self.split_on_predicate(sentence)
        original_sentence = " ".join(split_sentence)
        key = original_sentence if merge_repeated else sentence

        if original_sentence not in sentence_map:
            input_tokens = self.format_input(original_sentence)
            sentence_map[key] = {"input": input_tokens, "output": tf.zeros((len(input_tokens), len(BIO)))}
        else:
            input_tokens = sentence_map[key]["input"]

        token_sets = [self.tokenizer.encode(chunk, add_special_tokens=False) for chunk in split_sentence]
        predicate_start = len(token_sets[0]) + 1    # 1 is added to account for the [CLS] token that is added later
        predicate_end = predicate_start + len(token_sets[1])

        output_tags = [self.token_to_tag(t, i, predicate_start, predicate_end) for i, t in enumerate(input_tokens)]
        encoded_output_tags = tf.one_hot([tag.value for tag in output_tags], len(BIO))
        sentence_map[key]["output"] += encoded_output_tags

    def normalize_training_output(self, training_sentence_output: tf.Tensor, weighted=False):
        output = training_sentence_output
        if not weighted:
            # After add_training_sentence_data, if a token appears once as B and twice as O, its output might
            # look something like [1, 0, 2, 0], the first position representing B and the third representing O.
            # However, we are not interested in counting the number of appearences, just if there was an
            # appearence, so we clip the output, generating [1, 0, 1, 0].
            output = tf.clip_by_value(training_sentence_output, 0.0, 1.0)

        # The count_nonzero function returns a tensor with the number of different tags that were identified
        # for each token. Using the previous example, count_nonzero would return 2, as the token appears as
        # B and as O. What the repeat funcion then does is to repeat this 2 so it has the same shape as the
        # clipped output, so we can do a element-wise divison. Using the previous example, we would be able
        # to do [1, 0, 1, 0] / [2, 2, 2, 2] => [0.5, 0, 0.5, 0]).
        divisors = tf.math.reduce_sum(output, axis=1, keepdims=True)
        division_matrix = tf.repeat(divisors, len(BIO), axis=1)

        return output / division_matrix

    def format_training_data(self, training_sentences: list[str], merge_repeated=True,
                             weighted_merge=False) -> tuple[tf.Tensor, tf.Tensor]:
        sentence_map: SentenceMap = {}  # Keeps track of sentences that are repeated with different predicates
        for sentence in training_sentences:
            self.add_training_sentence_data(sentence, sentence_map, merge_repeated=merge_repeated)

        training_x = [v["input"] for v in sentence_map.values()]
        training_y = [self.normalize_training_output(v["output"], weighted_merge) for v in sentence_map.values()]
        return tf.stack(training_x), tf.stack(training_y)

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

    def print_elements(self, tokens: list[str], tags: list[BIO], scores: Union[list[float], None] = None):
        lines: list[list[str]] = []
        lines.append(tokens)
        lines.append([tag.name for tag in tags])
        if scores:
            score_chunks = []
            for score in scores:
                score_chunk = f"{score:.2} "
                score_chunk = score_chunk[1:] if score < 1.0 else score_chunk
                score_chunks.append(score_chunk)
            lines.append(score_chunks)

        chunk_lengths = [len(max(column, key=len)) for column in np.array(lines, dtype="object").transpose()]  # ignore
        for line in lines:
            for chunk, length in zip(line, chunk_lengths):
                print(chunk.ljust(length), end=" ")
            print()

    def print_annotated_sentence(self, sentence: str, sentence_output: tf.Tensor, o_threshold=0.0, show_scores=False):
        token_ids = self.tokenizer.encode(sentence)
        tokens: list[str] = self.tokenizer.convert_ids_to_tokens(token_ids)  # type: ignore
        formatted_sentence_output = self.format_output(sentence_output)

        tags = []
        scores = []
        for token_output in formatted_sentence_output[:len(tokens)]:
            tag, score = token_output[0]
            if tag == BIO.O and score < o_threshold:
                tag, score = token_output[1]
            tags.append(tag)
            scores.append(score)

        if show_scores:
            self.print_elements(tokens, tags, scores)
        else:
            self.print_elements(tokens, tags)

    def print_annotated_sentences(self, sentences: list[str], sentence_outputs: tf.Tensor,
                                  o_threshold=0.0, show_scores=False):
        for sentence, sentence_output in zip(sentences, sentence_outputs):
            self.print_annotated_sentence(sentence, sentence_output, o_threshold=o_threshold, show_scores=show_scores)
