import tensorflow as tf

from typing import Union

from .constants import DEFAULT_MODEL_NAME
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

    @classmethod
    def load(cls, name: str = DEFAULT_MODEL_NAME):
        return cls(name, name)

    def save(self, name: str = DEFAULT_MODEL_NAME):
        self.predicate_extractor.save(name)
        self.argument_predictor.save(name)

    def compile(self, pe_optimizer=None, pe_loss=None, pe_metrics=None, ap_optimizer=None, ap_loss=None,
                ap_metrics=None):
        default_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        default_loss = tf.keras.losses.CategoricalCrossentropy()
        default_metrics = [tf.keras.metrics.CategoricalCrossentropy()]

        pe_optimizer = pe_optimizer or ap_optimizer or default_optimizer
        pe_loss = pe_loss or ap_loss or default_loss
        pe_metrics = pe_metrics or ap_metrics or default_metrics

        ap_optimizer = ap_optimizer or pe_optimizer or default_optimizer
        ap_loss = ap_loss or pe_loss or default_loss
        ap_metrics = ap_metrics or pe_metrics or default_metrics

        self.predicate_extractor.compile(optimizer=pe_optimizer, loss=pe_loss, metrics=pe_metrics)
        self.argument_predictor.compile(optimizer=ap_optimizer, loss=ap_loss, metrics=ap_metrics)

    def summary(self):
        self.predicate_extractor.summary()
        self.argument_predictor.summary()

    def fit(self, training_sentences, *args, merge_repeated=False, epochs=20, pe_epochs=None, ap_epochs=None,
            early_stopping=False, callbacks=None, **kwargs):
        pe_epochs = pe_epochs or epochs
        ap_epochs = ap_epochs or epochs

        print("Predicate Extractor:")
        self.predicate_extractor.fit(training_sentences, *args, merge_repeated=merge_repeated, epochs=pe_epochs,
                                     early_stopping=early_stopping, callbacks=callbacks, **kwargs)
        print("\nArgument predictor:")
        self.argument_predictor.fit(training_sentences, *args, merge_repeated=merge_repeated, epochs=ap_epochs,
                                    early_stopping=early_stopping, callbacks=callbacks, **kwargs)

    def predict(self, sentences: list[str], pred_threshold=0.2, arg_threshold=0.15):
        arg_pred_inputs = self.predicate_extractor(sentences, acceptance_threshold=pred_threshold)
        outputs = self.argument_predictor(arg_pred_inputs, acceptance_threshold=arg_threshold)
        return outputs
