import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from dataset import load_data, get_tokens, load_or_get_tokens, convert_tokens_to_ints
from model import build_model

def train(batch_size, epochs):
    print('Loading data')
    data = load_data('data.csv')

    y = [d[1] for d in data]
    corpus = [d[0] for d in data]

    # Cache/load tokens
    tokens = load_or_get_tokens(corpus)

    X, vocab_size, largest_vector_len = convert_tokens_to_ints(tokens)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # TODO: Understand these lines of code
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=largest_vector_len)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=largest_vector_len)

    # For speed
    train_size = 1000
    test_size = 500
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]

    print('Training...')
    model = build_model(vocab_size, largest_vector_len)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('out/model.h5', monitor='acc', save_best_only=True, save_weights_only=True)
    ]

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

def logistic_regression():
    """
    Base case: logistic regression
    """
    data = load_data('data.csv')
    y = [d[1] for d in data]
    corpus = [d[0] for d in data]
    vectorizer = TfidfVectorizer(tokenizer=get_tokens)
    X = vectorizer.fit_transform(corpus)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lgs = LogisticRegression(solver='liblinear', verbose=1)
    lgs.fit(X_train, y_train)
    print('Test Accuracy: ', lgs.score(X_test, y_test))

def main():
    parser = argparse.ArgumentParser(description='Trains the model')
    parser.add_argument('--debug', default=False, type=bool, help='Debug mode')
    parser.add_argument('--batch-size', default=16, type=int, help='Size of training batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train for')
    args = parser.parse_args()

    if args.debug:
        # Turn on eager execution for debugging
        tf.enable_eager_execution()

    train(args.batch_size, args.epochs)

if __name__ == '__main__':
    main()
