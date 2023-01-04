import re
from transformers import AutoTokenizer
import tensorflow as tf

from .constants import BIO, PREDICATE_PATTERN, MAX_SENTENCE_SIZE


class DataFormatter():
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

    def split_on_predicate(self, sentence):
        split_sentence = re.split(PREDICATE_PATTERN, sentence)
        trimmed_split = [chunk.strip() for chunk in split_sentence]
        return trimmed_split

    def format_input(self, sentence):
        return self.tokenizer.encode(sentence, padding="max_length", max_length=MAX_SENTENCE_SIZE)

    def format_inputs(self, sentences):
        return tf.constant([self.format_input(sentence) for sentence in sentences])

    def format_output(self, output):
        predicted_tag_values = tf.math.argmax(output, 1) + 1
        predicted_tags = [BIO(prediction).name for prediction in predicted_tag_values]
        return predicted_tags

    def format_outputs(self, outputs):
        return [self.format_output(output) for output in outputs]

    def format_training_sentence(self, sentence):
        split_sentence = self.split_on_predicate(sentence)
        original_sentence = " ".join(split_sentence)

        token_sets = [self.tokenizer.encode(chunk, add_special_tokens=False) for chunk in split_sentence]
        predicate_start = len(token_sets[0]) + 1    # 1 is added to account for the [CLS] token that is added later
        predicate_end = predicate_start + len(token_sets[1])

        input_tokens = self.format_input(original_sentence)
        output_tags = [(BIO.B if i == predicate_start else BIO.I if predicate_start < i < predicate_end else BIO.O) for i, token in enumerate(input_tokens)]
        encoded_output_tags = tf.one_hot([tag.value - 1 for tag in output_tags], len(BIO))
        
        return input_tokens, encoded_output_tags

    def format_training_data(self, training_sentences):
        training_x = []
        training_y = []
        for sentence in training_sentences:
            x, y = self.format_training_sentence(sentence)
            training_x.append(x)
            training_y.append(y)
        return tf.stack(training_x), tf.stack(training_y)

    def print_annotated_sentence(self, sentence, tags):
        token_ids = self.tokenizer.encode(sentence)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        sentence_chunks = []
        annotation_chunks = []
        for token, tag in zip(tokens, tags):
            sentence_chunks.append(token)
            annotation_chunks.append(f"{tag:<{len(token)}}")

        token_sentence = " ".join(sentence_chunks)
        annotation = " ".join(annotation_chunks)
        
        print(token_sentence + "\n" + annotation)

    def print_annotated_sentences(self, sentences, outputs):
        tag_sets = self.format_outputs(outputs)
        for sentence, tags in zip(sentences, tag_sets):
            self.print_annotated_sentence(sentence, tags)
    