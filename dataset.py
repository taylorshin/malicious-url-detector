import numpy as np
import pandas as pd
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
            tokens_all = tokens_all + tokens_dot
    # Remove redundant tokens
    tokens_all = list(set(tokens_all))
    # Remove .com
    if 'com' in tokens_all:
        tokens_all.remove('com')

    return tokens_all

def main():
    data = load_data('data.csv')


if __name__ == '__main__':
    main()
