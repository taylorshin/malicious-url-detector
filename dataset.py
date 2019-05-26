import re
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

TOKEN_FNAME = 'out/tokens.pickle'

def load_data(filename):
    """
    Load data from file and return all URL data
    """
    df = pd.read_csv(filename)
    # Convert 'bad' (string) to 1 (int)
    # Malicious = 1, not malicious = 0
    df['label'] = df['label'].apply(lambda x: int(x == 'bad'))
    data = df.to_numpy()
    return data

def load_or_get_tokens(corpus):
    """
    Loads the tokens from a corpus
    """
    tokens = []

    if os.path.isfile(TOKEN_FNAME) and os.path.exists(TOKEN_FNAME):
        print('Token cache found.')
        with open(TOKEN_FNAME, 'rb') as f:
            tokens = pickle.load(f)
    else:
        print('Caching tokens...')
        tokens = [get_tokens(doc) for doc in corpus]
        # print('TOKENS: ', tokens)
        with open(TOKEN_FNAME, 'wb') as f:
            pickle.dump(tokens, f)

    return tokens

def get_tokens(url):
    """
    Tokenize the given URL
    """
    tokens = []
    # Split tokens initially by slash
    tokens_slash = url.split('/')
    tokens_slash = list(filter(None, tokens_slash))
    for ts in tokens_slash:
        tokens_dash = ts.split('-')
        for td in tokens_dash:
            tokens_dot = td.split('.')
            for tdot in tokens_dot:
                # Split joined words
                tokens_word = viterbi_segment(tdot)[0]
                # If word is split into more than 5 single characters, don't tokenize via viterbi method
                singles = [t for t in tokens_word if len(t) == 1]
                if len(singles) > 5:
                    tokens = tokens + tokens_dot
                    break
                else:
                    tokens = tokens + tokens_word

    # TODO: split by other symbols like =, ?, etc?

    # Remove redundant tokens
    # P: Do we want to? Should we not preserve the sequence?
    # tokens = list(set(tokens))

    # Remove .com
    # P: This may be useful in differentiating from less trust-worthy top-level domains. I'm getting slightly higher accuracy without.
    # if 'com' in tokens:
    #     tokens.remove('com')

    return tokens

def convert_tokens_to_ints(tokens):
    """
    Conver tokens to int sequences
    """
    vocab = set([token for doc in tokens for token in doc])
    vocab_size = len(vocab)
    token_dict = {token: i for i, token in enumerate(vocab)}

    largest_vector_len = 0
    for doc in tokens:
        if len(doc) > largest_vector_len:
            largest_vector_len = len(doc)

    int_seq = np.array([[token_dict[token] for token in doc] for doc in tokens])
    return int_seq, vocab_size, largest_vector_len

def viterbi_segment(text):
    """
    Taken from Stack Overflow: https://stackoverflow.com/questions/195010/how-can-i-split-multiple-joined-words
    Requires English dictionary from https://github.com/dwyl/english-words or similar
    """
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
    return words, probs[-1]

def word_prob(word):
    # This works because of global variables, but should adjust
    return dictionary[word] / TOTAL_NUM_WORDS

def words(text):
    return re.findall('[a-z]+', text.lower())

# TODO: Figure out how to get viterbi segment to work without these global variables
dictionary = Counter(words(open('words.txt').read()))
MAX_WORD_LENGTH = max(map(len, dictionary))
TOTAL_NUM_WORDS = float(sum(dictionary.values()))
