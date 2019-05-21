import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

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

def get_tokens(url):
    """
    Tokenize the given URL
    """
    tokens_all = []
    # Split tokens by slash
    tokens_slash = url.split('/')
    tokens_slash = list(filter(None, tokens_slash))
    # print('Tokens by slash: ', tokens_slash)
    for ts in tokens_slash:
        tokens_dash = ts.split('-')
        # print('Tokens by dash: ', tokens_dash)
        for td in tokens_dash:
            tokens_dot = td.split('.')
            # tokens_all = tokens_all + tokens_dot
            # Why not more for loops amiright?
            for tdot in tokens_dot:
                maybe_tokens = viterbi_segment(tdot)[0]
                tokens_all = tokens_all + maybe_tokens

    # Remove redundant tokens
    # P: Do we want to? Should we not preserve the sequence?
    # tokens_all = list(set(tokens_all))

    # Remove .com
    # P: This may be useful in differentiating from less trust-worthy top-level domains. I'm getting slightly higher accuracy without.
    # if 'com' in tokens_all:
    #     tokens_all.remove('com')

    return tokens_all

def viterbi_segment(text):
    """
    Taken from Stack Overflow: https://stackoverflow.com/questions/195010/how-can-i-split-multiple-joined-words
    Requires English dictionary from https://github.com/dwyl/english-words or similar
    """
    probs, lasts = [1.0], [0]
    for i in range(1, len(text) + 1):
        prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
                        for j in range(max(0, i - max_word_length), i))
        probs.append(prob_k)
        lasts.append(k)
    words = []
    i = len(text)
    while 0 < i:
        words.append(text[lasts[i]:i])
        i = lasts[i]
    words.reverse()
    return words, probs[-1]

def word_prob(word): return dictionary[word] / total # This works because of global variables, but should adjust

def words(text): return re.findall('[a-z]+', text.lower())

dictionary = Counter(words(open('words.txt').read()))
max_word_length = max(map(len, dictionary))
total = float(sum(dictionary.values()))
# End stealing from SO

def main():
    data = load_data('data.csv')


if __name__ == '__main__':
    main()
