import math
import tensorflow as tf

from typing import cast, Optional

from ..constants import DEFAULT_SENTENCE_SIZE, MODEL_DIR, PREDICATE_EXTRACTION_MODEL_DIR_NAME, DEFAULT_MODEL_NAME
from ..bert import bert
from .constants import ACCEPTANCE_THRESHOLD, O_THRESHOLD
from .data_formatter import DataFormatter

from .types import BIO, ArgPredInputs, PredicateMasks, SentenceIds, SentenceInputs


class PredicateExtractor(DataFormatter):
    def __init__(self, *dense_layer_units: int, name: Optional[str] = None,
                 sentence_size: int = DEFAULT_SENTENCE_SIZE) -> None:
        if name:
            self._load_model(name)
        else:
            super().__init__(sentence_size)
            self._config_model(*dense_layer_units)

    @staticmethod
    def full_model_path(name: str):
        return MODEL_DIR / name

    @staticmethod
    def pe_model_path(name: str):
        return PredicateExtractor.full_model_path(name) / PREDICATE_EXTRACTION_MODEL_DIR_NAME

    @classmethod
    def load(cls, name: str = DEFAULT_MODEL_NAME):
        if cls.pe_model_path(name).is_dir():
            return cls(name=name)
        else:
            raise Exception(f"Model {str} does not exist.")

    def _load_model(self, name: str):
        path = self.pe_model_path(name)
        if path.is_dir():
            model = tf.keras.models.load_model(path)
            self.model = cast(tf.keras.Model, model)
            self.sentence_size = self.model.layers[0].input_shape[0][1]
        else:
            raise Exception(f"Model {str} does not exist.")

    def _config_model(self, *dense_layer_units: int):
        token_ids = tf.keras.layers.Input(self.sentence_size, dtype="int32")
        embeddings = bert.encoder(token_ids)["last_hidden_state"]  # type: ignore

        dense_layers = embeddings
        for layer_units in dense_layer_units:
            dense_layers = tf.keras.layers.Dense(layer_units)(dense_layers)

        final_dense = tf.keras.layers.Dense(len(BIO))(dense_layers)
        softmax = tf.keras.layers.Softmax()(final_dense)

        self.model = tf.keras.Model(inputs=token_ids, outputs=softmax)
        self.model.layers[1].trainable = False

    def save(self, name: str = DEFAULT_MODEL_NAME):
        self.full_model_path(name).mkdir(exist_ok=True)
        path = self.pe_model_path(name)
        path.mkdir(exist_ok=True)
        self.model.save(path)

    def compile(self, optimizer=None, loss=None, metrics=None):
        optimizer = optimizer or tf.keras.optimizers.SGD(learning_rate=0.01)
        loss = loss or tf.keras.losses.CategoricalCrossentropy()
        metrics = metrics or [tf.keras.metrics.CategoricalCrossentropy()]

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        self.model.summary()

    def fit(self, training_sentences: list[str], *args, merge_repeated=False, epochs=20,
            early_stopping=False, callbacks=None, **kwargs):
        training_x, training_y = self.format_training_data(training_sentences, merge_repeated=merge_repeated)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=math.ceil(epochs / 10 if epochs > 100 else 10),
            min_delta=0.0003)  # type: ignore
        callbacks = callbacks or [early_stopping_callback] if early_stopping else []

        return self.model.fit(training_x, training_y, *args, epochs=epochs, callbacks=callbacks, **kwargs)

    def predict(self, inputs: SentenceInputs) -> tf.Tensor:
        return self.model.predict(inputs)

    def annotate_sentences(self, sentences: list[str], o_threshold=O_THRESHOLD, show_scores=False):
        inputs = self.format_inputs(sentences)
        outputs: tf.Tensor = self.predict(inputs)
        self.print_annotated_sentences(sentences, outputs, o_threshold=o_threshold, show_scores=show_scores)

    def __call__(self, sentences: list[str], acceptance_threshold=ACCEPTANCE_THRESHOLD) -> ArgPredInputs:
        inputs = self.format_inputs(sentences)
        outputs = self.predict(inputs)
        mask_sets = self.build_predicate_masks(outputs, acceptance_threshold=acceptance_threshold)

        sentence_ids: SentenceIds = []
        sentence_inputs: SentenceInputs = []
        masks: PredicateMasks = []

        current_id = 0
        for i, m_set in zip(inputs, mask_sets):
            for mask in m_set:
                sentence_ids.append(current_id)
                sentence_inputs.append(i)
                masks.append(mask)
            current_id += 1

        return sentence_ids, sentence_inputs, masks
