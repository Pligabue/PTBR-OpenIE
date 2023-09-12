from enum import Enum
import tensorflow as tf

from typing import TypedDict, Union

SentenceId = int
SentenceIds = list[SentenceId]
SentenceInput = list[int]
SentenceInputs = list[SentenceInput]
SentenceMapValue = TypedDict("SentenceMapValue", {"input": SentenceInput, "output": tf.Tensor, "count": int})
SentenceMap = dict[Union[str, int], SentenceMapValue]
ModelInput = list[SentenceInput]


class BIO(Enum):
    B = 0
    I = 1
    O = 2
    S = 3


TokenPrediction = tuple[BIO, float]
FormattedTokenOutput = list[TokenPrediction]
FormattedSentenceOutput = list[FormattedTokenOutput]

Variation = list[BIO]
SentenceVariations = list[Variation]
PredicateMask = list[bool]
PredicateMasks = list[PredicateMask]
ArgPredInputs = tuple[SentenceIds, SentenceInputs, PredicateMasks]
