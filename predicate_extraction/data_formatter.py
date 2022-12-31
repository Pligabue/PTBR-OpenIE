import yaml
from yaml import CLoader as Loader, CDumper as Dumper
import re
import tensorflow as tf
from transformers import AutoTokenizer

from .constants import BIO, PREDICATE_PREDICTION_DIR, PREDICATE_PATTERN, MAX_SENTENCE_SIZE

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

def split_on_predicate(sentence):
    split_sentence = re.split(PREDICATE_PATTERN, sentence)
    trimmed_split = [chunk.strip() for chunk in split_sentence]
    return trimmed_split

def format_input(sentence):
    return tokenizer.encode(sentence, padding="max_length", max_length=MAX_SENTENCE_SIZE)

def format_sentence(sentence):
    split_sentence = split_on_predicate(sentence)
    original_sentence = " ".join(split_sentence)

    token_sets = [tokenizer.encode(chunk, add_special_tokens=False) for chunk in split_sentence]
    predicate_start = len(token_sets[0]) + 1    # 1 is added to account for the [CLS] token that is added later
    predicate_end = predicate_start + len(token_sets[1])

    input_tokens = format_input(original_sentence)
    output_tags = [(BIO.B if i == predicate_start else BIO.I if predicate_start < i < predicate_end else BIO.O) for i, token in enumerate(input_tokens)]
    encoded_output_tags = tf.one_hot([tag.value - 1 for tag in output_tags], len(BIO))
    
    return {
        "sentence": original_sentence,
        "token_ids": input_tokens,
        "tokens": tokenizer.convert_ids_to_tokens(input_tokens),
        "predicate_span": (predicate_start, predicate_end),
        "output_tags": output_tags,
        "encoded_output_tags": encoded_output_tags
    }

def format_sentences(sentences):
    return [format_sentence(sentence) for sentence in sentences]

def format_output(output):
    predicted_tag_values = tf.math.argmax(output, 1) + 1
    predicted_tags = [BIO(prediction).name for prediction in predicted_tag_values]
    return predicted_tags

def print_annotated_sentence(sentence, tags):
    token_ids = tokenizer.encode(sentence)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    sentence_chunks = []
    annotation_chunks = []
    for i, token in enumerate(tokens):
        sentence_chunks.append(token)
        annotation_chunks.append(f"{tags[i]:<{len(token)}}")

    token_sentence = " ".join(sentence_chunks)
    annotation = " ".join(annotation_chunks)
    
    return token_sentence + "\n" + annotation

if __name__ == "__main__":
    with (PREDICATE_PREDICTION_DIR / "training_data.yml").open(encoding="utf-8") as f:
        sentences = yaml.load(f, Loader)
    format_sentences(sentences)