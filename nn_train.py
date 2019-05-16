import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import dataset

def train():
    print('Loading and tokenizing data...')
    data = dataset.load_data('data.csv')

    y = [d[1] for d in data]
    corpus = [d[0] for d in data]
    vectorizer = TfidfVectorizer(tokenizer=dataset.get_tokens)
    X = vectorizer.fit_transform(corpus)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Training...')
    model = construct_model(X_train.shape[1])
    model.fit(X_train, y_train, batch_size=32, epochs=1, verbose=5)

    print('Evaluating...')
    score, acc = model.evaluate(X_test, y_test, verbose=2, batch_size=32)
    print('Score: %.2f' % score)
    print('Validation accuracy: %.2f' % acc)

def construct_model(input_length):
    # For now, following tutorial at: https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47
    embed_dim = 128
    lstm_out = 200
    batch_size = 32
    tf.logging.set_verbosity(tf.logging.ERROR)

    model = Sequential()
    model.add(Embedding(2500, embed_dim,input_length=input_length))
    model.add(LSTM(lstm_out, dropout=0.2))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model

def main():
    train()

if __name__ == '__main__':
    main()