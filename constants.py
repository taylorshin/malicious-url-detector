import os

OUT_DIR = 'out'
LOG_DIR = os.path.join(OUT_DIR, 'logs')
MODEL_FILE = os.path.join(OUT_DIR, 'model.h5')
LOSS_PLOT_FILE = os.path.join(OUT_DIR, 'loss.png')
ACC_PLOT_FILE = os.path.join(OUT_DIR, 'acc.png')
TOKEN_FNAME = os.path.join(OUT_DIR, 'tokens.pkl')
