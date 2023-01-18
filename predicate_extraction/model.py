from transformers import TFAutoModel
import tensorflow as tf
import math

from .constants import BIO, MAX_SENTENCE_SIZE, O_THRESHOLD
from .data_formatter import DataFormatter

class PredicateExtractor(DataFormatter):
    def __init__(self, *dense_layer_units: int) -> None:
        super().__init__()

        bert = TFAutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

        token_ids = tf.keras.layers.Input(MAX_SENTENCE_SIZE, dtype="int32")
        embeddings = bert(token_ids)["last_hidden_state"]
        
        dense_layers = embeddings
        for layer_units in dense_layer_units:
            dense_layers = tf.keras.layers.Dense(layer_units)(dense_layers)

        final_dense = tf.keras.layers.Dense(len(BIO))(dense_layers)
        softmax = tf.keras.layers.Softmax()(final_dense)
        
        self.model = tf.keras.Model(inputs=token_ids, outputs=softmax)
        self.model.layers[1].trainable = False

    def compile(self, optimizer=None, loss=None, metrics=None):
        optimizer = optimizer or tf.keras.optimizers.SGD(learning_rate=0.02)
        loss = loss or tf.keras.losses.CategoricalCrossentropy()
        metrics = metrics or [tf.keras.metrics.CategoricalCrossentropy()]
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def summary(self):
        self.model.summary()

    def fit(self, training_sentences, *args, epochs=20, early_stopping=False, callbacks=None, **kwargs):
        training_x, training_y = self.format_training_data(training_sentences)
        
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=math.ceil(epochs/10 if epochs > 100 else 10),
            min_delta=0.0003)
        callbacks = callbacks or [early_stopping_callback] if early_stopping else []

        return self.model.fit(training_x, training_y, *args, epochs=epochs, callbacks=callbacks, **kwargs)

    def predict(self, sentences, o_threshold=O_THRESHOLD, show_scores=False):
        inputs = self.format_inputs(sentences)
        outputs: tf.Tensor = self.model.predict(inputs)
        self.print_annotated_sentences(sentences, outputs, o_threshold=o_threshold, show_scores=show_scores)
