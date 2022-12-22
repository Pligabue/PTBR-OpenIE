import yaml
from yaml import CLoader as Loader, CDumper as Dumper
import re
from transformers import AutoTokenizer

from .constants import MAX_SENTENCE_SIZE, PREDICATE_PREDICTION_DIR, PREDICATE_PATTERN

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

def split_on_predicate(sentence):
    split_sentence = re.split(PREDICATE_PATTERN, sentence)
    trimmed_split = [chunk.strip() for chunk in split_sentence]
    return trimmed_split

def format_sentence(sentence):
    split_sentence = split_on_predicate(sentence)
    token_sets = [tokenizer.encode(chunk, add_special_tokens=False) for chunk in split_sentence]
    predicate_start = len(token_sets[0]) + 1    # 1 is added to account for the [CLS] token that is added later
    predicate_end = predicate_start + len(token_sets[1])
    input_tokens = [101] + [token for token_set in token_sets for token in token_set] + [102]
    output_tags = [("P-B" if i == predicate_start else "P-I" if predicate_start < i < predicate_end else "P-O") for i, token in enumerate(input_tokens)]
    
    return {
        "token_ids": input_tokens,
        "tokens": tokenizer.convert_ids_to_tokens(input_tokens),
        "predicate_span": (predicate_start, predicate_end),
        "output_tags": output_tags
    }

def format_data(sentences):
    format_sentence(sentences[0])

if __name__ == "__main__":
    with (PREDICATE_PREDICTION_DIR / "training_data.yml").open(encoding="utf-8") as f:
        sentences = yaml.load(f, Loader)
    format_data(sentences)