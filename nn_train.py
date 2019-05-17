import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import dataset

def train():
    print('Loading and tokenizing data...')
    data = dataset.load_data('data.csv')

    y = [d[1] for d in data]
    corpus = [d[0] for d in data]

    # vectorizer = TfidfVectorizer(tokenizer=dataset.get_tokens)
    # X = vectorizer.fit_transform(corpus)

    tokens = [dataset.get_tokens(doc) for doc in corpus]
    X, vocab_size, largest_vector_len = tokens_to_int_sequence(tokens)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = sequence.pad_sequences(X_train, maxlen=largest_vector_len)
    X_test = sequence.pad_sequences(X_test, maxlen=largest_vector_len)

    # For speed
    train_size = 1000
    test_size = 500
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]

    print('Training...')
    model = construct_model(vocab_size, largest_vector_len)
    model.fit(X_train, y_train, batch_size=32, epochs=1, verbose=5)

    print('Evaluating...')
    score, acc = model.evaluate(X_test, y_test, verbose=2, batch_size=32)
    print('Score: %.2f' % score)
    print('Validation accuracy: %.2f' % acc)

def construct_model(vocab_size, largest_vector_len):
    # Resources:
    # https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47
    # https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

    # Play with these
    embedding_output_dim = 2
    lstm_units = 1

    tf.logging.set_verbosity(tf.logging.ERROR)

    model = Sequential()
    # model.add(Embedding(embedding_input_dim, embedding_output_dim))
    # model.add(LSTM(lstm_units, dropout=0.2)) # play with dropout
    # # model.add(Dense(1, activation='softmax')) # could use default activation
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.add(Embedding(vocab_size, embedding_output_dim, input_length=largest_vector_len))
    model.add(LSTM(lstm_units))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

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

def main():
    train()
    # evaluating should probably be a separate function or something...

if __name__ == '__main__':
    main()