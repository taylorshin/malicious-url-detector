import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from dataset import load_data, load_or_get_tokens, convert_tokens_to_ints, get_tokens
from util import build_or_load_model
from constants import MODEL_FILE

def evaluate(model, X, y):
    score, acc = model.evaluate(X, y, verbose=2)#, batch_size=16)
    print('Score: %.2f' % score)
    print('Validation accuracy: %.2f%%' % (acc * 100))

def encode_url_for_prediction(url):
    full_data = load_data('data.csv')
    y = [d[1] for d in full_data]
    corpus = [d[0] for d in full_data]
    tokens = load_or_get_tokens(corpus)
    X, vocab_size, largest_vector_len, token_dict = convert_tokens_to_ints(tokens)
    # print('num data points: ', X.shape)
    # print('largest_vector_len:', largest_vector_len)

    url_tokens = get_tokens(url)
    # print('Tokens:', url_tokens)
    # int_seq, vocab_size, max_vector_length = convert_tokens_to_ints(tokens)

    int_seq = []
    for token in url_tokens:
        if token in token_dict:
            int_seq.append(token_dict[token])
        else:
            int_seq.append(0) # UNK token
    int_seq = [int_seq]

    # print('Int Seq:', int_seq)
    padded = tf.keras.preprocessing.sequence.pad_sequences(int_seq, maxlen=largest_vector_len) # TODO: be better
    # print('Padded Vectors:', padded)
    return padded[0]

def main():
    parser = argparse.ArgumentParser(description='Make predictions from trained model')
    parser.add_argument('--model', default=MODEL_FILE, type=str, help='Path to model file')
    args = parser.parse_args()

    # Load the test data
    data = load_data('data.csv')
    y = [d[1] for d in data]
    corpus = [d[0] for d in data]

    print('Dist of data:', np.mean(y))

    # Cache/load tokens
    tokens = load_or_get_tokens(corpus)
    X, vocab_size, largest_vector_len, _ = convert_tokens_to_ints(tokens)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use pad_sequences to standardize the lengths
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=largest_vector_len)
    test_size = int(X_test.shape[0] / 8)
    print('Test data size: {}'.format(test_size))
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]

    print('Some test data:')
    print(X_test[:5])
    print(y_test[:5])

    # Load the model
    model = build_or_load_model(args.model, vocab_size, largest_vector_len)
    
    print('Evaluating...')
    evaluate(model, X_test, y_test)
    print()

    test_urls = [
        'defsnotspam.biz/',
        'google.com',
        '174vjaijskjrkjviw4.co.ts',
        'stackoverflow.com/questions/17394882/add-dimensions-to-a-numpy-array',
        'cambuihostel.com/tmp/chase/7b2592844f7cc97a0f4150e7a7de3a36/',
    ]

    for url in test_urls:
        print('Test URL: ', url)
        pred_seq = np.array(encode_url_for_prediction(url))
        pred_seq = pred_seq[np.newaxis, ...]
        # print('Prediction Sequence Going In:', pred_seq)
        # print('Sequence size:', pred_seq.shape)
        pred = model.predict(pred_seq, batch_size=1, verbose=1)
        print('Prediction value: ', pred[0][0])
        print()


    print('\n\n\n\nExample prediction: custom')
    prediction_seq = np.array([1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 1, 553, 746, 21, 743, 426, 2351, 452, 1])
    prediction_seq = prediction_seq[np.newaxis, ...]
    # print('Prediction Sequence Going In:', prediction_seq)
    # print('Sequence size:', prediction_seq.shape)
    prediction = model.predict(prediction_seq, batch_size=1, verbose=1)

    print('Prediction value:', prediction[0][0])
    if prediction[0][0] < .5:
        print('Eh, it\'s probably fine!')
    else:
        print('Be wary, traveler.')
    

    print('\n\n\n\nExample prediction: test data 1')
    prediction_seq = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 188804, 251606,   1463,  51190,
  231499, 255416, 211142, 209901])
    prediction_seq = prediction_seq[np.newaxis, ...]
    # print('Prediction Sequence Going In:', prediction_seq)
    # print('Sequence size:', prediction_seq.shape)
    prediction = model.predict(prediction_seq, batch_size=1, verbose=1)

    print('Prediction value:', prediction[0][0])
    if prediction[0][0] < .5:
        print('Eh, it\'s probably fine!')
    else:
        print('Be wary, traveler.')



    print('\n\n\n\nExample prediction: test data 1')
    prediction_seq = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
,  0, 0,  40597, 209901,  52869, 186725, 271923, 274684,  13017,  51190,
  179116,   8835, 108020, 169832])
    prediction_seq = prediction_seq[np.newaxis, ...]
    # print('Prediction Sequence Going In:', prediction_seq)
    # print('Sequence size:', prediction_seq.shape)
    prediction = model.predict(prediction_seq, batch_size=1, verbose=1)

    print('Prediction value:', prediction[0][0])
    if prediction[0][0] < .5:
        print('Eh, it\'s probably fine!')
    else:
        print('Be wary, traveler.')


if __name__ == '__main__':
    main()
