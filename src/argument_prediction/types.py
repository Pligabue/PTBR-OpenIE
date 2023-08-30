from typing import TypedDict, Union
import tensorflow as tf

TrainingX = list[tf.Tensor]
TrainingY = tf.Tensor
TrainingData = tuple[TrainingX, TrainingY]
SentenceMapValue = TypedDict("SentenceMapValue", {"output": tf.Tensor, "count": int})
SentenceMap = dict[Union[str, int], SentenceMapValue]
Span = tuple[int, int]


class BIO(Enum):
    SB = 0
    SI = 1
    OB = 2
    OI = 3
    O = 4
    S = 5
