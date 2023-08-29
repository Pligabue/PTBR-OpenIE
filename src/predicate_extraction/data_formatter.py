import re
from transformers import AutoTokenizer
import tensorflow as tf

from .constants import PREDICATE_PATTERN, MAX_SENTENCE_SIZE, SPECIAL_TOKEN_IDS
from .types import BIO, SentenceMap, SentenceInput, FormattedTokenOutput, FormattedSentenceOutput


class DataFormatter():
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

    def split_on_predicate(self, sentence) -> list[str]:
        split_sentence = re.split(PREDICATE_PATTERN, sentence)
        trimmed_split = [chunk.strip() for chunk in split_sentence if chunk]
        return trimmed_split

    def format_input(self, sentence) -> SentenceInput:
        return self.tokenizer.encode(sentence, padding="max_length", max_length=MAX_SENTENCE_SIZE)

    def format_inputs(self, sentences) -> tf.Tensor:
        return tf.constant([self.format_input(sentence) for sentence in sentences])

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

    def print_annotated_sentence(self, sentence: str, sentence_output: tf.Tensor, o_threshold=0.0, show_scores=False):
        token_ids = self.tokenizer.encode(sentence)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        formatted_sentence_output = self.format_output(sentence_output)

        sentence_chunks = []
        annotation_chunks = []
        score_chunks = []
        for token, token_output in zip(tokens, formatted_sentence_output):
            tag, score = token_output[0]
            if tag == BIO.O and score < o_threshold:
                tag, score = token_output[1]

            rounded_score = f"{score:.2} " if show_scores else ""
            rounded_score = rounded_score[1:] if rounded_score.startswith("0") else rounded_score

            length = max(len(token), len(rounded_score)) if show_scores else len(token)

            sentence_chunks.append(f"{token:<{length}}")
            annotation_chunks.append(f"{tag.name:<{length}}")
            score_chunks.append(f"{rounded_score:<{length}}")

        token_sentence = " ".join(sentence_chunks)
        annotation = " ".join(annotation_chunks)
        scores = " ".join(score_chunks)

        annotated_sentence = "\n".join([token_sentence, annotation, scores])
        print(annotated_sentence) if show_scores else print(token_sentence + "\n" + annotation)

    def print_annotated_sentences(self, sentences: list[str], sentence_outputs: tf.Tensor,
                                  o_threshold=0.0, show_scores=False):
        for sentence, sentence_output in zip(sentences, sentence_outputs):
            self.print_annotated_sentence(sentence, sentence_output, o_threshold=o_threshold, show_scores=show_scores)