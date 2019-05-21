import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def build_model(vocab_size, largest_vector_len):
    # Resources:
    # https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47
    # https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

    # Play with these
    embedding_output_dim = 2
    lstm_units = 8

    # tf.logging.set_verbosity(tf.logging.ERROR)

    model = tf.keras.Sequential()
    # model.add(Embedding(embedding_input_dim, embedding_output_dim))
    # model.add(LSTM(lstm_units, dropout=0.2)) # play with dropout
    # # model.add(Dense(1, activation='softmax')) # could use default activation
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.add(layers.Embedding(vocab_size, embedding_output_dim, input_length=largest_vector_len))
    model.add(layers.LSTM(lstm_units))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
