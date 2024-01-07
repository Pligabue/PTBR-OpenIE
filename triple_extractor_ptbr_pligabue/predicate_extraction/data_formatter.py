import itertools
import re
import numpy as np
import tensorflow as tf

from typing import Union

from ..bert import bert
from ..constants import (DEFAULT_SENTENCE_SIZE, OBJECT_PATTERN, PREDICATE_PATTERN,
                         SPECIAL_TOKEN_IDS, SUBJECT_PATTERN)

from .types import (BIO, SentenceMap, SentenceMapValue, SentenceInput, SentenceInputs,
                    FormattedTokenOutput, FormattedSentenceOutput, Variation, SentenceVariations,
                    PredicateMask, PredicateMasks)


class DataFormatter():
    def __init__(self, sentence_size=DEFAULT_SENTENCE_SIZE):
        self.sentence_size = sentence_size

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
        return bert.tokenizer.encode(sentence, padding="max_length", truncation=True, max_length=self.sentence_size)

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

    def add_training_sentence_data(self, sentence: str, sentence_map: SentenceMap, merge_repeated=False):
        split_sentence = self.split_on_predicate(sentence)
        original_sentence = " ".join(split_sentence).strip()
        key = original_sentence if merge_repeated else sentence

        if key not in sentence_map:
            input_tokens = self.format_input(original_sentence)
            initial_output = tf.zeros((len(input_tokens), len(BIO)))
            sentence_map[key] = {"input": input_tokens, "output": initial_output, "count": 0}
        else:
            input_tokens = sentence_map[key]["input"]

        token_sets = [bert.tokenizer.encode(chunk, add_special_tokens=False) for chunk in split_sentence]
        predicate_start = len(token_sets[0]) + 1    # 1 is added to account for the [CLS] token that is added later
        predicate_end = predicate_start + len(token_sets[1])

        output_tags = [self.token_to_tag(t, i, predicate_start, predicate_end) for i, t in enumerate(input_tokens)]
        encoded_output_tags = tf.one_hot([tag.value for tag in output_tags], len(BIO))
        sentence_map[key]["output"] += encoded_output_tags
        sentence_map[key]["count"] += 1

    def normalize_training_output(self, sentence_map: SentenceMapValue, weighted=False):
        if not weighted:
            clipped = tf.clip_by_value(sentence_map["output"], 0.0, 1.0)
            normalization_matrix = tf.math.reduce_sum(clipped, axis=1, keepdims=True)
            return clipped / normalization_matrix
        return sentence_map["output"] / sentence_map["count"]  # type: ignore

    def format_training_data(self, training_sentences: list[str], merge_repeated=True,
                             weighted_merge=False) -> tuple[tf.Tensor, tf.Tensor]:
        sentence_map: SentenceMap = {}  # Keeps track of sentences that are repeated with different predicates
        for sentence in training_sentences:
            self.add_training_sentence_data(sentence, sentence_map, merge_repeated=merge_repeated)

        training_x = [v["input"] for v in sentence_map.values()]
        training_y = [self.normalize_training_output(v, weighted_merge) for v in sentence_map.values()]
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
        token_ids = bert.tokenizer.encode(sentence)
        tokens: list[str] = bert.tokenizer.convert_ids_to_tokens(token_ids)  # type: ignore
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

    ###############################
    # ARGUMENT PREDICTION SECTION #
    ###############################

    def get_filtered_sentence_output(self, sentence_output: FormattedSentenceOutput,
                                     acceptance_threshold=0.2) -> FormattedSentenceOutput:
        filtered_sentence_output = []
        for token_output in sentence_output:
            tags_above_threshold = [pred for pred in token_output if pred[1] > acceptance_threshold]
            if len(tags_above_threshold) == 0:
                tags_above_threshold.append(token_output[0])
                # If no tag is above the threshold, take the one with
                # the hightest score (the list is sorted).
            filtered_sentence_output.append(tags_above_threshold)

        return filtered_sentence_output

    def addition_is_valid(self, sequence: Variation, tag: BIO) -> bool:
        if not sequence:
            return tag == BIO.S

        if BIO.B in sequence and tag == BIO.B:
            return False

        last_tag = sequence[-1]
        if last_tag == BIO.B or last_tag == BIO.I:
            return tag == BIO.O or tag == BIO.I
        if last_tag == BIO.O or last_tag == BIO.S:
            return tag != BIO.I
        return True

    def sequence_is_valid(self, sequence: Variation) -> bool:
        has_predicate = BIO.B in sequence
        has_padding = sequence[0] == BIO.S and sequence[-1] == BIO.S
        correct_length = len(sequence) == self.sentence_size
        return has_predicate and has_padding and correct_length

    def build_sentence_variations(self, sentence_output: FormattedSentenceOutput, acceptance_threshold=0.2):
        filtered_sentence_output = self.get_filtered_sentence_output(sentence_output, acceptance_threshold)
        variations: SentenceVariations = [[]]
        for token_output in filtered_sentence_output:
            tags = [prediction[0] for prediction in token_output]
            combinations = itertools.product(variations, tags)
            variations = [var + [tag] for var, tag in combinations if self.addition_is_valid(var, tag)]

        valid_variations = [variation for variation in variations if self.sequence_is_valid(variation)]
        return valid_variations

    def build_variations(self, sentence_outputs: tf.Tensor, acceptance_threshold=0.2) -> list[SentenceVariations]:
        formatted_sentence_outputs = self.format_outputs(sentence_outputs)
        variations: list[SentenceVariations] = []
        for formatted_sentence_output in formatted_sentence_outputs:
            sentence_variations = self.build_sentence_variations(formatted_sentence_output, acceptance_threshold)
            variations.append(sentence_variations)

        return variations

    def get_superset(self, mask_a: PredicateMask, mask_b: PredicateMask):
        a_start = mask_a.index(True)
        a_end = a_start + mask_a.count(True)
        b_start = mask_b.index(True)
        b_end = b_start + mask_b.count(True)

        if (a_start <= b_start and a_end > b_end) or (a_start < b_start and a_end >= b_end):
            return mask_a
        if (b_start <= a_start and b_end > a_end) or (b_start < a_start and b_end >= a_end):
            return mask_b
        return None

    def replace_subsets(self, masks: PredicateMasks):
        n_masks = len(masks)
        replaced_masks = masks[:]
        for i in range(n_masks):
            for j in range(i + 1, n_masks):
                mask = replaced_masks[i]
                other_mask = replaced_masks[j]
                if superset_mask := self.get_superset(mask, other_mask):
                    replaced_masks[i] = superset_mask
                    replaced_masks[j] = superset_mask
        return replaced_masks

    def remove_repeated(self, masks: PredicateMasks) -> PredicateMasks:
        mask_map: dict[str, PredicateMask] = {}
        for mask in masks:
            key = "".join([str(v) for v in mask])
            mask_map[key] = mask
        return mask_map.values()  # type: ignore

    def build_predicate_masks(self, sentence_outputs: tf.Tensor, acceptance_threshold=0.2):
        sentence_variation_sets = self.build_variations(sentence_outputs, acceptance_threshold)
        target_tags = (BIO.B, BIO.I)
        mask_sets: list[PredicateMasks] = []
        for sentence_variations in sentence_variation_sets:
            masks: PredicateMasks = []
            for variation in sentence_variations:
                masks.append([tag in target_tags for tag in variation])
            masks = self.replace_subsets(masks)
            masks = self.remove_repeated(masks)
            mask_sets.append(masks)
        return mask_sets
