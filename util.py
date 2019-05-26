import numpy as np
import tensorflow as tf

def build_or_load_model(model_dir, vocab_size, largest_vector_len, allow_load=True):
    from model import build_model
    model = build_model(vocab_size, largest_vector_len)
    # model[0].summary()
    if allow_load:
        try:
            model[0].load_weights(MODEL_FILE)
            print('Loaded model from file.')
        except:
            print('Unable to load model from file.')
    return model
