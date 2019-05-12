import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filename):
    """
    Load data from file and split into train and test sets
    """
    df = pd.read_csv(filename)
    # Convert 'bad' (string) to 1 (int)
    # Malicious = 1, not malicious = 0
    df['label'] = df['label'].apply(lambda x: int(x == 'bad'))
    data = df.to_numpy()

    # Train test split
    print(data[:3, 0])
    print(data[:3, 1])    
    X_train, X_test, y_train, y_test = train_test_split(data[:, 0], data[:, 1], test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def get_tokens(url):
    """
    Tokenize the given URL
    """
    tokens_all = []
    # Split tokens by slash
    tokens_slash = url.split('/')
    tokens_slash = list(filter(None, tokens_slash))
    print('Tokens by slash: ', tokens_slash)
    for ts in tokens_slash:
        tokens_dash = ts.split('-')
        print('Tokens by dash: ', tokens_dash)
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
    X_train, X_test, y_train, y_test = load_data('data.csv')
    print('X_train: ', X_train.shape)
    print('X_test: ', X_test.shape)
    print('y_train: ', y_train.shape)
    print('y_test: ', y_test.shape)
    test = X_train[0]
    print('Test URLs: ', test)
    tokens = get_tokens(test)
    print('TOKENS: ', tokens)


if __name__ == '__main__':
    main()
