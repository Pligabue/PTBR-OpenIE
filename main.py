import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Registers the ops.

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="sentence")
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", name="tokenizer")
encoder_inputs = preprocessor(text_input)

encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/2", trainable=True, name="bert_embedding")
outputs = encoder(encoder_inputs)
# pooled_output = outputs["pooled_output"]      # [batch_size, 512].
sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 512].

self_attention = tf.keras.layers.Attention(name="self_attention")
self_attention_output = self_attention([sequence_output, sequence_output, sequence_output])

label_classifier = tf.keras.layers.Dense(4, name="label_classifier")
output = label_classifier(self_attention_output)

model = tf.keras.Model(inputs=text_input, outputs=output)
model.summary()

n_of_extractions = 1
sentences = ["Obamas was born in Hawaii."]

