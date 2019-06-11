# Imports
from collections import Counter
import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def viterbi_segment(text):
    """
    Taken from Stack Overflow: https://stackoverflow.com/questions/195010/how-can-i-split-multiple-joined-words
    Requires English dictionary from https://github.com/dwyl/english-words or similar
    """
    split_url = re.split(r'[\/\-\.\&\?\=\_]+', text)
    results = np.array([])
    
    for text in split_url:
        probs, lasts = [1.0], [0]
        for i in range(1, len(text) + 1):
            prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
                            for j in range(max(0, i - MAX_WORD_LENGTH), i))
            probs.append(prob_k)
            lasts.append(k)
        words = []
        i = len(text)
        while 0 < i:
            words.append(text[lasts[i]:i])
            i = lasts[i]
        words.reverse()

        singles = np.sum([1 if len(word) == 1 else 0 for word in words])
        if singles == len(words):    # If split completely by character, put back together
            result = ''.join(words)
        elif singles > 3:            # If number of single characters > 3, put back together
            result = ''.join(words)
        else:                        # Otherwise join with periods to tokenize
            result = '.'.join(words)
        results = np.append(results, [result])

    return '.'.join(results)

def word_prob(word):
    return dictionary[word] / TOTAL_NUM_WORDS

def words(text):
    return re.findall('[a-z]+', text.lower())

dictionary = Counter(words(open('../mobydick.txt').read()))
MAX_WORD_LENGTH = max(map(len, dictionary))
TOTAL_NUM_WORDS = float(sum(dictionary.values()))



vocab = set()
vocab_size = 0
token_dict = {}
largest_vector_length = 0

def convert_tokens_to_ints(tokens, max_features, max_tokens=-1, training=True):
    """
    Convert tokens to int sequences
    """

    global vocab, vocab_size, token_dict, largest_vector_length

    if training:
        token_dict = dict({token: i for i, token in enumerate([token for doc in tokens for token in doc])})
        vocab_size = len(token_dict)

        largest_vector_length = 0
        for doc in tokens:
            if len(doc) > largest_vector_length:
                largest_vector_length = len(doc)

    int_seq = []
    for doc in tokens:
        doc_seq = []
        for token in doc:
            if token not in token_dict:
                vocab.add(token)
                token_dict[token] = vocab_size
                vocab_size += 1
            
            doc_seq.append(token_dict[token] % max_features)

        doc_seq = np.array(doc_seq)
        int_seq.append(doc_seq)
    
    if max_tokens == -1:
        x = pad_sequences(int_seq, largest_vector_length)
    else:
        x = pad_sequences(int_seq, max_tokens, truncating='post')

    return np.stack(x)