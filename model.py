import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers

def build_model(vocab_size, largest_vector_len):
    # Resources:
    # https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47
    # https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

    # Play with these
    embedding_output_dim = 16
    lstm_units = 128

    tf.logging.set_verbosity(tf.logging.ERROR)

    model = tf.keras.Sequential()

    model.add(layers.Embedding(vocab_size, embedding_output_dim, input_length=largest_vector_len))
    model.add(layers.LSTM(lstm_units, dropout=0.7, recurrent_dropout=0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
