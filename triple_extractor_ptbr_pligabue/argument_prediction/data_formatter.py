import itertools
import re
import tensorflow as tf
from typing import Union

from ..predicate_extraction.types import SentenceInput, SentenceInputs, PredicateMasks
from ..bert import bert
from ..constants import (DEFAULT_SENTENCE_SIZE, OBJECT_PATTERN, PREDICATE_PATTERN,
                         SPECIAL_TOKEN_IDS, SUBJECT_PATTERN)

from .types import (BIO, TrainingData, FormattedSentenceOutput, FormattedTokenOutput, Mask,
                    Masks, SubjectMask, SubjectMasks, ObjectMask, ObjectMasks, SubjectMaskSets,
                    ObjectMaskSets, MaskSets, SentenceMap, SentenceMapValue, SentenceVariations,
                    Span, Variation)


class DataFormatter():
    def __init__(self, sentence_size=DEFAULT_SENTENCE_SIZE):
        self.sentence_size = sentence_size

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
        token_sets: list = bert.tokenizer(split_sentence, add_special_tokens=False)["input_ids"]  # type: ignore
        start = len(token_sets[0]) + 1  # 1 is added to account for the [CLS] token that is added later
        end = start + len(token_sets[1])
        return (start, end)

    def span_to_mask(self, span):
        start, finish = span
        return [start <= i < finish for i in range(self.sentence_size)]

    def tokenize(self, sen: Union[list[str], str]):
        if isinstance(sen, list):
            return bert.tokenizer(sen, padding="max_length", max_length=self.sentence_size)["input_ids"]
        return bert.tokenizer.encode(sen, padding="max_length", max_length=self.sentence_size)

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

    ###########
    # TRIPLES #
    ###########

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

        if BIO.SB in sequence and tag == BIO.SB:
            return False
        if BIO.OB in sequence and tag == BIO.OB:
            return False

        last_tag = sequence[-1]
        if last_tag == BIO.SB or last_tag == BIO.SI:
            return tag == BIO.O or tag == BIO.SI
        if last_tag == BIO.OB or last_tag == BIO.OI:
            return tag == BIO.O or tag == BIO.OI
        if last_tag == BIO.O or last_tag == BIO.S:
            return tag != BIO.SI and tag != BIO.OI
        return True

    def sequence_is_valid(self, sequence: Variation) -> bool:
        has_subject = BIO.SB in sequence
        has_object = BIO.OB in sequence
        has_padding = sequence[0] == BIO.S and sequence[-1] == BIO.S
        correct_length = len(sequence) == self.sentence_size
        return has_subject and has_object and has_padding and correct_length

    def build_sentence_variations(self, sentence_output: FormattedSentenceOutput, acceptance_threshold=0.2):
        filtered_sentence_output = self.get_filtered_sentence_output(sentence_output, acceptance_threshold)
        variations: SentenceVariations = [[]]
        for token_output in filtered_sentence_output:
            tags = [pred[0] for pred in token_output]
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

    def get_superset(self, mask_a: Mask, mask_b: Mask):
        a_start = mask_a.index(True)
        a_end = a_start + mask_a.count(True)
        b_start = mask_b.index(True)
        b_end = b_start + mask_b.count(True)

        if (a_start <= b_start and a_end > b_end) or (a_start < b_start and a_end >= b_end):
            return mask_a
        if (b_start <= a_start and b_end > a_end) or (b_start < a_start and b_end >= a_end):
            return mask_b
        return None

    def replace_subsets(self, masks: Masks):
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

    def remove_repeated(self, subj_masks: SubjectMasks, obj_masks: ObjectMasks) -> tuple[SubjectMasks, ObjectMasks]:
        mask_map: dict[str, tuple[SubjectMask, ObjectMask]] = {}
        for subj_mask, obj_mask in zip(subj_masks, obj_masks):
            key = "".join([str(v) for v in subj_mask]) + "".join([str(v) for v in obj_mask])
            mask_map[key] = subj_mask, obj_mask
        new_subj_masks = [v[0] for v in mask_map.values()]
        new_obj_masks = [v[1] for v in mask_map.values()]
        return new_subj_masks, new_obj_masks  # type: ignore

    def build_masks(self, sentence_outputs: tf.Tensor, acceptance_threshold=0.2) -> MaskSets:
        sentence_variation_sets = self.build_variations(sentence_outputs, acceptance_threshold)
        subject_target_tags = (BIO.SB, BIO.SI)
        object_target_tags = (BIO.OB, BIO.OI)

        sentence_subject_mask_sets: SubjectMaskSets = []
        sentence_object_mask_sets: ObjectMaskSets = []
        for sentence_variations in sentence_variation_sets:
            subject_masks: SubjectMasks = []
            object_masks: ObjectMasks = []
            for variation in sentence_variations:
                subject_masks.append([tag in subject_target_tags for tag in variation])
                object_masks.append([tag in object_target_tags for tag in variation])
            subject_masks = self.replace_subsets(subject_masks)
            object_masks = self.replace_subsets(object_masks)
            subject_masks, object_masks = self.remove_repeated(subject_masks, object_masks)
            sentence_subject_mask_sets.append(subject_masks)
            sentence_object_mask_sets.append(object_masks)
        return sentence_subject_mask_sets, sentence_object_mask_sets

    def fix_partial_words(self, tokens: list[str], mask: Mask):
        fixed_mask: Mask = mask[:]

        n_tokens = len(tokens)
        for i in range(1, n_tokens):
            previous_token_is_part_of_mask = fixed_mask[i-1]
            current_token_is_part_of_mask = fixed_mask[i]
            current_token_is_part_of_previous_word = tokens[i].startswith("##")
            current_token_should_be_part_of_mask = (previous_token_is_part_of_mask
                                                    and not current_token_is_part_of_mask
                                                    and current_token_is_part_of_previous_word)
            if current_token_should_be_part_of_mask:
                fixed_mask[i] = True

        for i in range(n_tokens - 2, 0, -1):
            current_token_is_part_of_mask = fixed_mask[i]
            next_token_is_part_of_mask = fixed_mask[i+1]
            next_token_is_part_of_current_word = tokens[i+1].startswith("##")
            current_token_should_be_part_of_mask = (not current_token_is_part_of_mask
                                                    and next_token_is_part_of_mask
                                                    and next_token_is_part_of_current_word)
            if current_token_should_be_part_of_mask:
                fixed_mask[i] = True

        return fixed_mask
