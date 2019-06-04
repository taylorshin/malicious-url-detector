import numpy as np
import tensorflow as tf

def build_or_load_model(model_dir, vocab_size, largest_vector_len, allow_load=True):
    from model import build_model
    model = build_model(vocab_size, largest_vector_len)
    model.summary()
    if allow_load:
        try:
            model.load_weights(model_dir)
            print('Loaded model from file.')
        except Exception as e:
            print('Unable to load model from file.')
            print(str(e))
    return model
