import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from dataset import load_data, load_or_get_tokens, convert_tokens_to_ints
from util import build_or_load_model

MODEL_DIR = 'out/model.h5'

def evaluate(model, X, y):
    score, acc = model.evaluate(X, y, verbose=2)#, batch_size=16)
    print('Score: %.2f' % score)
    print('Validation accuracy: %.2f%%' % (acc * 100))

def main():
    parser = argparse.ArgumentParser(description='Make predictions from trained model')
    parser.add_argument('--model', default=MODEL_DIR, type=str, help='Path to model file')
    args = parser.parse_args()

    # Load the test data
    data = load_data('data.csv')
    y = [d[1] for d in data]
    corpus = [d[0] for d in data]

    # Cache/load tokens
    tokens = load_or_get_tokens(corpus)
    X, vocab_size, largest_vector_len = convert_tokens_to_ints(tokens)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use pad_sequences to standardize the lengths
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=largest_vector_len)
    test_size = 500
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]

    # Load the model
    model = build_or_load_model(args.model, vocab_size, largest_vector_len)
    
    print('Evaluating...')
    evaluate(model, X_test, y_test)


if __name__ == '__main__':
    main()
