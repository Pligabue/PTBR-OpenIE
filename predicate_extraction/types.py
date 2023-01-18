from enum import EnumMeta
from typing import Tuple

SentenceInput = list[int]
ModelInput = list[SentenceInput]
FormattedTokenOutput = list[Tuple[EnumMeta, float]]
FormattedSentenceOutput = list[FormattedTokenOutput]