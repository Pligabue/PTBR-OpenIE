from pathlib import Path
from enum import Enum, EnumMeta

MAX_SENTENCE_SIZE = 100
PREDICATE_PREDICTION_DIR = Path(__file__).parent
PREDICATE_PATTERN = r"\[(.*)\]"
BIO: EnumMeta = Enum("BIO", ["B", "I", "O", "S"], start=0)
SPECIAL_TOKEN_IDS = [101, 102, 0]
O_THRESHOLD = 0.8