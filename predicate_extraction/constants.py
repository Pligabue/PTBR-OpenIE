from pathlib import Path
from enum import Enum

MAX_SENTENCE_SIZE = 50
PREDICATE_PREDICTION_DIR = Path(__file__).parent
PREDICATE_PATTERN = r"\[(.*)\]"
BIO = Enum("BIO", ["B", "I", "O"])