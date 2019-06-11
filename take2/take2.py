# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

# Imports
import pickle
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

from take2_utils import *
from take2_constants import *

tf.logging.set_verbosity(tf.logging.ERROR)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Droid Serif']



# Read data
print('Reading data...')
if os.path.isfile(VIT_PICKLE_FNAME) and os.path.exists(VIT_PICKLE_FNAME):
    with open(VIT_PICKLE_FNAME, 'rb') as f:
        data = pickle.load(f)      # (n_samples, 2)
else:
    data = pd.read_csv(DATA_FNAME) # (n_samples, 2)

    # Preprocess data
    print('Applying Viterbi algorithm...')
    data['url'] = data['url'].apply(viterbi_segment)
    data = data.to_numpy()
    with open(VIT_PICKLE_FNAME, 'wb') as f:
        pickle.dump(data, f)

x = data[:, 0] # (n_samples)
y = data[:, 1] # (n_samples)

print('Generating one-hot vectors...')
x = [url.split('.') for url in x]
x = convert_tokens_to_ints(x, max_features, max_tokens=max_num_tokens)
y = np.array([1 if label == 'bad' else 0 for label in y])

# Build model
print('Building model...')
model = Sequential()
model.add(Embedding(max_features, output_dim=embedding_output_dim))
model.add(LSTM(lstm_units))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, stratify=y, random_state=97)
x_train = x_train[:int(x_train.shape[0] * training_fraction)]
y_train = y_train[:int(y_train.shape[0] * training_fraction)]

# Train model
print('Training model...')
# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=val_split)

train_accuracies = []
train_losses = []
val_accuracies = []
val_losses = []
rskf = RepeatedStratifiedKFold(n_splits=kfolds_splits, n_repeats=kfolds_repeats)
for train_index, test_index in rskf.split(x_train, y_train):
    history = model.fit(x_train[train_index], y_train[train_index], batch_size=batch_size, epochs=epochs, validation_data=(x_train[test_index], y_train[test_index]))
    train_accuracies.append(history.history['acc'])
    train_losses.append(history.history['loss'])
    val_accuracies.append(history.history['val_acc'])
    val_losses.append(history.history['val_loss'])

train_accuracies = np.array(train_accuracies)
train_losses = np.array(train_losses)
val_accuracies = np.array(val_accuracies)
val_losses = np.array(val_losses)

# Save model
print('Saving model...')
model.save(MODEL_FNAME)

# Evaluate model
print('Evaluating model...')
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Evaluation:',  score)

train_accuracies = train_accuracies.flatten()
train_losses = train_losses.flatten()
val_accuracies = val_accuracies.flatten()
val_losses = val_losses.flatten()

# Plot training & validation accuracy values
plt.plot(train_accuracies, color='#cccccc')
plt.plot(val_accuracies, color='#1280ae')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(ACC_PLOT_FNAME)
plt.show()

# Plot training & validation loss values
plt.figure()
plt.plot(train_losses, color='#cccccc')
plt.plot(val_losses, color='#1280ae')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(LOSS_PLOT_FNAME)
plt.show()
