import re
import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from constants import TOKEN_FNAME
import re

# Global dictionary of top level domains
TLDs = {}

def load_data(filename):
    """
    Load data from file and return all URL data
    """
    print('Loading data...')
    df = pd.read_csv(filename)

    # Convert 'bad' (string) to 1 (int). Malicious = 1, benign = 0
    df['label'] = df['label'].apply(lambda x: int(x == 'bad'))

    # Balance dataset to 50% benign and 50% malicious
    good_count, bad_count = df.label.value_counts().tolist()
    df = df.drop(df[df['label'] == 0].sample(n=good_count-bad_count).index)

    return df.to_numpy()

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
        os.makedirs(os.path.dirname('out/'), exist_ok=True)
        with open(TOKEN_FNAME, 'wb') as f:
            pickle.dump(tokens, f)

    return tokens

def get_tokens(url):
    """
    Tokenize the given URL
    """
    tokens = re.split(r'[\/\-\.\&\?\=\_]+', url)
    for i, token in enumerate(tokens):
        word_split = viterbi_segment(token)[0]
        if len(word_split) < 4:
            tokens[i:i+1] = word_split[0:len(word_split)]
    return tokens

    # tokens = []
    # # Split tokens initially by slash
    # tokens_slash = url.split('/')
    # tokens_slash = list(filter(None, tokens_slash))
    # for ts in tokens_slash:
    #     tokens_dash = ts.split('-')
    #     for td in tokens_dash:
    #         tokens_dot = td.split('.')
    #         for tdot in tokens_dot:
    #             # Split joined words
    #             tokens_word = viterbi_segment(tdot)[0]
    #             # If word is split into more than 5 single characters, don't tokenize via viterbi method
    #             singles = [t for t in tokens_word if len(t) == 1]
    #             if len(singles) > 5:
    #                 tokens = tokens + tokens_dot
    #                 break
    #             else:
    #                 tokens = tokens + tokens_word
    # return tokens

def extract_features(url):
    url_length = len(url)
    n_digits = sum(c.isdigit() for c in url)

    slash_loc = url.find('/')
    dot_loc = url[:slash_loc].rfind('.')
    top_domain = url[dot_loc+1:slash_loc]
    if top_domain not in TLDs:
        TLDs[top_domain] = len(TLDs)
    top_domain = TLDs[top_domain]

    return url_length, n_digits, top_domain

def convert_tokens_to_ints(tokens):
    """
    Convert tokens to int sequences
    """
    vocab = set([token for doc in tokens for token in doc])
    vocab_size = len(vocab)
    token_dict = {token: i for i, token in enumerate(vocab)}

    largest_vector_len = 0
    for doc in tokens:
        if len(doc) > largest_vector_len:
            largest_vector_len = len(doc)

    int_seq = []
    for doc in tokens:
        doc_seq = []
        for token in doc:
            doc_seq.append(token_dict[token])
        int_seq.append(doc_seq)

    return np.array(int_seq), vocab_size, largest_vector_len, token_dict

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

def main():
    # data = load_data('data.csv')

    # Save dictionary for website use
    # with open('dict_count.json', 'w') as f:
    #     f.write(json.dumps(dictionary))
    
    # word_split = viterbi_segment('helloworld')[0]
    # print('Word split:', word_split)

    print(get_tokens('realinnovation.com/css/menu.js'))

if __name__ == '__main__':
    main()
