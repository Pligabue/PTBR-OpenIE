from enum import Enum
from typing import Tuple, TypedDict, Union
import tensorflow as tf

SentenceInput = list[int]
SentenceMapValue = TypedDict("SentenceMapValue", {"input": SentenceInput, "output": tf.Tensor})
SentenceMap = dict[Union[str, int], SentenceMapValue]
ModelInput = list[SentenceInput]


class BIO(Enum):
    B = 0
    I = 1
    O = 2
    S = 3


FormattedTokenOutput = list[Tuple[BIO, float]]
FormattedSentenceOutput = list[FormattedTokenOutput]
