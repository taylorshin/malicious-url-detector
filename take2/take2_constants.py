# Constants and Parameters
DATA_FNAME = '../data.csv'
DICT_FNAME = '../mobydick.txt'
VIT_PICKLE_FNAME = 'vit-data-mobydick.pkl'

MODEL_FNAME = 'take2-model.h5'
LOSS_PLOT_FNAME = 'take2-loss.png'
ACC_PLOT_FNAME = 'take2-accuracy.png'

training_fraction = 0.2
test_split = 0.1
kfolds_splits = 5
kfolds_repeats = 3
epochs = 1

max_num_tokens = 32
max_features = 8192 # 1024
embedding_output_dim = 256
lstm_units = 128
dropout = 0.5
batch_size = 16
