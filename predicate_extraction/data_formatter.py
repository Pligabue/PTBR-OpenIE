import re
from transformers import AutoTokenizer
import tensorflow as tf

from .constants import BIO, PREDICATE_PATTERN, MAX_SENTENCE_SIZE, SPECIAL_TOKEN_IDS
from .types import *

class DataFormatter():
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

    def split_on_predicate(self, sentence) -> list[str]:
        split_sentence = re.split(PREDICATE_PATTERN, sentence)
        trimmed_split = [chunk.strip() for chunk in split_sentence]
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

    def format_training_sentence(self, sentence) -> Tuple[SentenceInput, tf.Tensor]:
        split_sentence = self.split_on_predicate(sentence)
        original_sentence = " ".join(split_sentence)

        token_sets = [self.tokenizer.encode(chunk, add_special_tokens=False) for chunk in split_sentence]
        predicate_start = len(token_sets[0]) + 1    # 1 is added to account for the [CLS] token that is added later
        predicate_end = predicate_start + len(token_sets[1])

        input_tokens = self.format_input(original_sentence)
        output_tags = [self.token_to_tag(token, i, predicate_start, predicate_end) for i, token in enumerate(input_tokens)]
        encoded_output_tags = tf.one_hot([tag.value for tag in output_tags], len(BIO))
        
        return input_tokens, encoded_output_tags

    def format_training_data(self, training_sentences: list[str]) -> Tuple(tf.Tensor, tf.Tensor):
        training_x: list[SentenceInput] = []
        training_y: list[tf.Tensor] = []
        for sentence in training_sentences:
            x, y = self.format_training_sentence(sentence)
            training_x.append(x)
            training_y.append(y)
        return tf.stack(training_x), tf.stack(training_y)

    def print_annotated_sentence(self, sentence: str, sentence_output: tf.Tensor, o_threshold=0.0, show_scores=False):
        token_ids = self.tokenizer.encode(sentence)
        tokens: list[str] = self.tokenizer.convert_ids_to_tokens(token_ids)
        formatted_sentence_output = self.format_output(sentence_output)
        
        sentence_chunks = []
        annotation_chunks = []
        score_chunks = []
        for token, token_output in zip(tokens, formatted_sentence_output):
            tag, score = token_output[0]
            if tag == BIO.O and score < o_threshold:
                tag, score = token_output[1]
            
            rounded_score = f"{score:.2} " if show_scores else None
            rounded_score = rounded_score[1:] if rounded_score.startswith("0") else rounded_score

            length = max(len(token), len(rounded_score)) if show_scores else len(token)

            sentence_chunks.append(f"{token:<{length}}")
            annotation_chunks.append(f"{tag.name:<{length}}")
            score_chunks.append(f"{rounded_score:<{length}}")

        token_sentence = " ".join(sentence_chunks)
        annotation = " ".join(annotation_chunks)
        scores = " ".join(score_chunks)
        
        print(token_sentence + "\n" + annotation + "\n" + scores) if show_scores else print(token_sentence + "\n" + annotation)

    def print_annotated_sentences(self, sentences: list[str], sentence_outputs: tf.Tensor, o_threshold=0.0, show_scores=False):
        for sentence, sentence_output in zip(sentences, sentence_outputs):
            self.print_annotated_sentence(sentence, sentence_output, o_threshold=o_threshold, show_scores=show_scores)
    