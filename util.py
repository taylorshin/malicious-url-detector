import os
import pickle
import numpy as np
from dataset import get_tokens

TOKEN_FNAME = 'tokens.bin'

def load_tokens(corpus):
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
        with open(TOKEN_FNAME, 'wb') as f:
            pickle.dump(tokens, f)

    return tokens
