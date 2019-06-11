import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers

def build_model(vocab_size, largest_vector_len, emb_dim=64, lstm_units=128, lr=1e-4, dropout_rate=0.5):
    """
    1D convolution and LSTM coming soon
    """
    tf.logging.set_verbosity(tf.logging.ERROR)

    # model = tf.keras.Sequential()

    # # Embedding layer
    # model.add(layers.Embedding(vocab_size, emb_dim, input_length=largest_vector_len))
    # model.add(layers.Dropout(dropout_rate))

    # # Convolutional layer
    # model.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    # model.add(layers.MaxPooling1D(pool_size=2))

    # # LSTM layer
    # model.add(layers.LSTM(lstm_units, kernel_regularizer=regularizers.l2(0.05), activity_regularizer=regularizers.l1(0.05))) # , recurrent_dropout=dropout_rate)) # input_shape=(largest_vector_len, 4)))
    # # model.add(layers.Dropout(dropout_rate))

    # # TODO: Try some of the examples here: https://keras.io/getting-started/sequential-model-guide/

    # # Output layer
    # model.add(layers.Dense(1, activation='sigmoid'))

    model = tf.keras.Sequential()
    model.add(layers.Embedding(1024, output_dim=256))
    model.add(layers.LSTM(128))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
