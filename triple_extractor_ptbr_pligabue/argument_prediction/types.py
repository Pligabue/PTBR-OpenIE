from enum import Enum
from typing import TypedDict, Union
import tensorflow as tf

from ..predicate_extraction.types import SentenceId, SentenceInput, PredicateMask

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


TokenPrediction = tuple[BIO, float]
FormattedTokenOutput = list[TokenPrediction]
FormattedSentenceOutput = list[FormattedTokenOutput]

Variation = list[BIO]
SentenceVariations = list[Variation]
Mask = list[bool]
Masks = list[Mask]
SubjectMask = Mask
SubjectMasks = list[SubjectMask]
SubjectMaskSets = list[SubjectMasks]
ObjectMask = Mask
ObjectMasks = list[ObjectMask]
ObjectMaskSets = list[ObjectMasks]
MaskSets = tuple[SubjectMaskSets, ObjectMaskSets]

ArgPredOutput = tuple[SentenceId, SentenceInput, PredicateMask, SubjectMask, ObjectMask]
ArgPredOutputs = list[ArgPredOutput]
