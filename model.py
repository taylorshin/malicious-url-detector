import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers

def build_model(vocab_size, largest_vector_len, emb_dim=128, lstm_units=128, lr=1e-3, dropout_rate=0.5):
    """
    1D convolution and LSTM coming soon
    """
    tf.logging.set_verbosity(tf.logging.ERROR)

    model = tf.keras.Sequential()

    # Embedding layer
    model.add(layers.Embedding(vocab_size, emb_dim, input_length=largest_vector_len))
    model.add(layers.Dropout(dropout_rate))

    # Convolutional layer
    # TODO: Experiment with conv layer

    # LSTM layer
    model.add(layers.LSTM(lstm_units, recurrent_dropout=dropout_rate)) # input_shape=(largest_vector_len, 4)))
    model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
