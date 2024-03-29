import tensorflow as tf
import math

from typing import cast, Optional

from ..constants import DEFAULT_SENTENCE_SIZE, MODEL_DIR, ARGUMENT_PREDICTION_MODEL_DIR_NAME, DEFAULT_MODEL_NAME
from ..bert import bert
from ..predicate_extraction.types import ArgPredInputs

from .constants import N_HEADS, ACCEPTANCE_THRESHOLD
from .types import BIO, ArgPredOutputs
from .data_formatter import DataFormatter


class ArgumentPredictor(DataFormatter):
    def __init__(self, *layer_units: int, name: Optional[str] = None,
                 sentence_size: int = DEFAULT_SENTENCE_SIZE) -> None:
        if name:
            self._load_model(name)
        else:
            super().__init__(sentence_size)
            self._config_model(*layer_units)

    @staticmethod
    def full_model_path(name: str):
        return MODEL_DIR / name

    @staticmethod
    def ap_model_path(name: str):
        return ArgumentPredictor.full_model_path(name) / ARGUMENT_PREDICTION_MODEL_DIR_NAME

    @classmethod
    def load(cls, name: str = DEFAULT_MODEL_NAME):
        return cls(name=name)

    def _load_model(self, name: str):
        path = self.ap_model_path(name)
        if path.is_dir():
            model = tf.keras.models.load_model(path)
            self.model = cast(tf.keras.Model, model)
            self.sentence_size = self.model.layers[0].input_shape[0][1]
        else:
            raise Exception(f"Model {str} does not exist.")

    def _config_model(self, *layer_units: int):
        lstm_units = layer_units[0]
        dense_layer_units = layer_units[1:]

        token_ids = tf.keras.layers.Input(self.sentence_size, dtype="int32")
        mask = tf.keras.layers.Input(self.sentence_size, dtype="bool")

        base_embeddings = bert.encoder(token_ids)["last_hidden_state"]  # type: ignore
        predicate_embeddings = tf.ragged.boolean_mask(base_embeddings, mask)
        mean_pred_embedding = tf.math.reduce_mean(predicate_embeddings, axis=1)
        mean_pred_as_matrix = tf.expand_dims(mean_pred_embedding, axis=1)
        mean_pred_tiled = tf.repeat(mean_pred_as_matrix, self.sentence_size, axis=1)

        mask_as_matrix = tf.keras.layers.Reshape([self.sentence_size, 1])(mask)
        n_repetions = 1 + N_HEADS - (mean_pred_tiled.shape[-1] * 2 + 1) % N_HEADS
        mask_as_matrix = tf.repeat(mask_as_matrix, n_repetions, axis=2)
        mask_as_matrix = tf.cast(mask_as_matrix, float)

        full_embeddings = tf.keras.layers.Concatenate(axis=2)([base_embeddings, mean_pred_tiled, mask_as_matrix])

        forward_layer = tf.keras.layers.LSTM(lstm_units, return_sequences=True)   # ignore
        backward_layer = tf.keras.layers.LSTM(lstm_units, return_sequences=True, go_backwards=True)  # ignore
        bi_lstm = tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode="ave")
        bi_lstm_output = bi_lstm(full_embeddings)

        dense_layers = bi_lstm_output
        for units in dense_layer_units:
            dense_layers = tf.keras.layers.Dense(units)(dense_layers)

        final_dense = tf.keras.layers.Dense(len(BIO))(dense_layers)
        softmax = tf.keras.layers.Softmax()(final_dense)

        self.model = tf.keras.Model(inputs=[token_ids, mask], outputs=softmax)
        self.model.layers[2].trainable = False

    def save(self, name: str):
        self.full_model_path(name).mkdir(exist_ok=True)
        path = self.ap_model_path(name)
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

    def predict(self, inputs: ArgPredInputs) -> tf.Tensor:
        _, sentences, predicate_masks = inputs
        model_inputs = [tf.constant(sentences), tf.constant(predicate_masks)]
        return self.model.predict(model_inputs)

    def __call__(self, inputs: ArgPredInputs, acceptance_threshold=ACCEPTANCE_THRESHOLD,
                 fix_partial_words=True) -> ArgPredOutputs:
        n_predicates = len(inputs[0])

        if n_predicates == 0:
            return []

        outputs: tf.Tensor = self.predict(inputs)  # type: ignore
        subject_mask_sets, object_mask_sets = self.build_masks(outputs, acceptance_threshold=acceptance_threshold)

        triples: ArgPredOutputs = []
        results = zip(*inputs, subject_mask_sets, object_mask_sets)
        for sentence_id, tokens, predicate_mask, sub_mask_set, obj_mask_set in results:
            token_strings: list[str] = bert.tokenizer.convert_ids_to_tokens(tokens)  # type: ignore
            for sub_mask, obj_mask in zip(sub_mask_set, obj_mask_set):
                triple = (sentence_id, tokens, predicate_mask, sub_mask, obj_mask)
                if self.clean_triple(triple, token_strings):
                    if fix_partial_words:
                        self.fix_partial_words(token_strings, predicate_mask)
                        self.fix_partial_words(token_strings, sub_mask)
                        self.fix_partial_words(token_strings, obj_mask)
                    triples.append(triple)
        return triples
