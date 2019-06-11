# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

# Imports
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import load_model

from take2_utils import *
from take2_constants import *

tf.logging.set_verbosity(tf.logging.ERROR)



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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, stratify=y, random_state=97)
x_train = x_train[:int(x_train.shape[0] * training_fraction)]
y_train = y_train[:int(y_train.shape[0] * training_fraction)]

print('Evaluating model...')
model = load_model(MODEL_FNAME)
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Evaluation:', score)

def prepare_single_for_input(url):
    viterbi_string = viterbi_segment(url)
    x = [viterbi_string.split('.')]
    x = convert_tokens_to_ints(x, max_features, max_tokens=max_num_tokens, training=False)

    return x

def prepare_many_for_input(urls):
    viterbi_strings = [viterbi_segment(url) for url in urls]
    x = [url.split('.') for url in viterbi_strings]
    x = convert_tokens_to_ints(x, max_features, max_tokens=max_num_tokens, training=False)

    return x

urls = [
    'google.com',
    'taylorshin.github.io',
    'pwschaedler.github.io',
    'stackoverflow.com/questions/29438265/stratified-train-test-split-in-scikit-learn',
    'vox.com/policy-and-politics/2019/6/5/18653800/trump-approval-rating-by-state-2020-election-odds',
    'vvellsfargo.com/bank'
]
input_sequences = prepare_many_for_input(urls)
predictions = model.predict(input_sequences)

for i, prediction in enumerate(predictions):
    print('URL:', urls[i], '\tProbability:', prediction, '=>', 'bad' if prediction[0] > 0.5 else 'good')
