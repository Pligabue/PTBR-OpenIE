from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf

from .constants import MAX_SENTENCE_SIZE


def build_model():
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    bert = TFAutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

    token_ids = tf.keras.layers.Input(MAX_SENTENCE_SIZE, dtype="int32")
    embeddings = bert(token_ids)["last_hidden_state"]
    feed_forward = tf.keras.layers.Dense(32)(embeddings)
    softmax = tf.keras.layers.Softmax()(feed_forward)
    model = tf.keras.Model(inputs=token_ids, outputs=softmax)
    model.layers[1].trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalCrossentropy()],
    )

    return model
