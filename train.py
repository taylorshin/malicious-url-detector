import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from dataset import load_data, get_tokens
from model import build_model
from util import load_tokens

def train():
    print('Loading and tokenizing data...')
    data = load_data('data.csv')

    y = [d[1] for d in data]
    corpus = [d[0] for d in data]

    # vectorizer = TfidfVectorizer(tokenizer=dataset.get_tokens)
    # X = vectorizer.fit_transform(corpus)

    # Cache/load tokens
    tokens = load_tokens(corpus)

    X, vocab_size, largest_vector_len = tokens_to_int_sequence(tokens)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

    model.fit(X_train, y_train, batch_size=32, epochs=1, callbacks=callbacks)

def tokens_to_int_sequence(tokens):
    vocab = set([token for doc in tokens for token in doc])
    vocab_size = len(vocab)
    dic = {token: i for i, token in enumerate(vocab)}

    largest_vector_len = 0
    for doc in tokens:
        if len(doc) > largest_vector_len:
            largest_vector_len = len(doc)

    int_seq = np.array([[dic[token] for token in doc] for doc in tokens])
    return int_seq, vocab_size, largest_vector_len

def logistic_regression():
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
    args = parser.parse_args()

    train()

if __name__ == '__main__':
    main()
