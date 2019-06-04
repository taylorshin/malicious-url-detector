import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from dataset import load_data, get_tokens, load_or_get_tokens, convert_tokens_to_ints # , extract_features
from model import build_model
from constants import LOSS_PLOT_FILE, ACC_PLOT_FILE, MODEL_FILE, LOG_DIR, OUT_DIR

def train(batch_size, epochs, lr, dropout_rate, model_file=MODEL_FILE):
    data = load_data('data.csv')

    # Labels
    y = [d[1] for d in data]
    corpus = [d[0] for d in data]

    # Cache/load tokens
    tokens = load_or_get_tokens(corpus)

    X, vocab_size, largest_vector_len, _ = convert_tokens_to_ints(tokens)
    # X_features = [extract_features(url) for url in corpus]

    # Use pad_sequences to standardize the lengths
    print('Padding sequences and extracting features...')
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=largest_vector_len)

    # some kind of zip between X (list of lists of ints) and X_features (list of tuples)
    # X = np.array([[(token, X_features[i][0], X_features[i][1], X_features[i][2]) for token in X[i]] for i in range(len(X))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split train into validation and train. Final split for train, val, test is 60%, 20%, 20%
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Use partial dataset for speed
    # train_size = int(X_train.shape[0] / 32)
    # val_size = int(X_val.shape[0] / 32)
    # print('Training data size: {}'.format(train_size))
    # print('Validation data size: {}'.format(val_size))
    # X_train = X_train[:train_size]
    # y_train = y_train[:train_size]
    # X_val = X_val[:val_size]
    # y_val = y_val[:val_size]

    print('Training...')
    print('Training data shape:', X_train.shape)
    model = build_model(vocab_size, largest_vector_len, lr=lr, dropout_rate=dropout_rate)
    model.summary()
    
    # TODO: figure out whether to monitor ACC or LOSS
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_file, monitor='loss', save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=0)
    ]

    return model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, callbacks=callbacks)

def logistic_regression():
    """
    Base case for classification
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

    # Train the model
    history = train(args.batch_size, args.epochs)

    ### Plot training and validation loss over epochs ###
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(train_loss))
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='red')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(LOSS_PLOT_FILE)

    plt.figure()
    plt.plot(history.history['acc'], color='blue')
    plt.plot(history.history['val_acc'], color='red')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(ACC_PLOT_FILE)

    """
    ### Hyperparameter search ###
    lrs = [1e-3, 1e-4, 1e-5]
    # emb_dim_list = [16, 32, 64, 128, 256]
    # emb_dim_list = [32, 64, 128]
    # lstm_units_list = [16, 32, 64, 128, 256]
    # lstm_units_list = [32, 64, 128]
    dropout_rate_list = [0.25, 0.5, 0.75]

    train_losses = []
    val_losses = []

    for lr in lrs:
        for dropout_rate in dropout_rate_list:
            print('LR: {}, DR: {}'.format(lr, dropout_rate))
            param_str = 'lr' + str(lr) + '_dr' + str(dropout_rate)
            # Train the model
            model_file = os.path.join(OUT_DIR, 'model_' + param_str + '.h5')
            history = train(args.batch_size, args.epochs, lr, dropout_rate, model_file)
            train_losses.append(history.history['loss'][-1])
            val_losses.append(history.history['val_loss'][-1])
            print('Train losses: ', train_losses)
            print('Val losses: ', val_losses)
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs = range(len(train_loss))
            # Plot and save
            plt.figure()
            plt.plot(epochs, train_loss, label='Training Loss', color='blue')
            plt.plot(epochs, val_loss, label='Validation Loss', color='red')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plot_file = os.path.join(OUT_DIR, 'loss_' + param_str + '.png')
            plt.savefig(plot_file)
            print('-----------------------------------------------------------------------')
    """


if __name__ == '__main__':
    main()
