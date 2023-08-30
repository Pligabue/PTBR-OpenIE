import tensorflow as tf

from typing import Union

from .argument_prediction import ArgumentPredictor
from .predicate_extraction import PredicateExtractor
from .data_formatter import DataFormatter


class TripleExtractor(DataFormatter):
    def __init__(self, pe_layers_or_name: Union[tuple[int], str], ap_layers_or_name: Union[tuple[int], str]) -> None:
        if isinstance(pe_layers_or_name, str):
            self.predicate_extractor = PredicateExtractor.load(pe_layers_or_name)
        else:
            self.predicate_extractor = PredicateExtractor(*pe_layers_or_name)

        if isinstance(ap_layers_or_name, str):
            self.argument_predictor = ArgumentPredictor.load(ap_layers_or_name)
        else:
            self.argument_predictor = ArgumentPredictor(*ap_layers_or_name)

        super().__init__(self.predicate_extractor.tokenizer)
