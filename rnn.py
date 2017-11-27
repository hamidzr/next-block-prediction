import numpy as np
np.random.seed(42)
import itertools
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Sequential, load_model
from keras.preprocessing.text import one_hot, text_to_word_sequence
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
from pylab import rcParams
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 12, 5

INPUT = './data/cleanSample.txt'
WORD2VEC_MODEL = './data/block2vec100.model'
RESULTS_DIR = './results'
START_SYMBOL = 'START'
END_SYMBOL = 'END'

# memoize helper
def memoize(f):
    cache = {}
    def decorated(*args):
        key = (f, str(args))
        result = cache.get(key, None)
        if result is None:
            result = f(*args)
            cache[key] = result
        return result
    return decorated

# pad the scripts with begining and ending
def pad_script(script):
    # add seq_size -1 to the begining of the script
    script.append(END_SYMBOL)
    for _ in range(0, SEQ_SIZE-1):
        script.insert(0, START_SYMBOL)
    return script

def load_data(num_scripts=100000, padding=True):
    tokens_list = []
    with open(INPUT, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            line_words = text_to_word_sequence(line)
            if (padding): line_words = pad_script(line_words)
            tokens_list.append(line_words)
            if (idx >= num_scripts): break
    all_tokens = itertools.chain.from_iterable(tokens_list)
    word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
    # convert token lists to token-id lists, e.g. [[1, 2], [2, 2]] here
    token_ids = [[word_to_id[token] for token in tokens_doc] for tokens_doc in tokens_list]
    return token_ids, word_to_id

@memoize
def one_hot_encode(number):
    vec = np.zeros(VOCAB_SIZE, dtype=bool)
    np.put(vec, number, True)
    return vec

def encode_dataset(x, encoder):
    encodedSet = []
    for seq in x:
        encoded = list(map(encoder, seq))
        encodedSet.append(encoded)
    return encodedSet


# split a list randomly into two with the given ratio
def split_data(aList, ratio=0.8):
    np.random.shuffle(aList)
    splitPoint = int(ratio * len(aList))
    return aList[0:splitPoint], aList[splitPoint:]

# build lists of sequences and the actual next word for that response
def build_xy(scripts_list):
    sequences = []
    next_words = []
    for script in scripts_list:
        scriptSize = len(script)
        for idx in range(0, scriptSize-SEQ_SIZE):
            sequences.append(script[idx:idx+SEQ_SIZE])
            next_words.append(script[idx+SEQ_SIZE])
    return sequences, next_words

SEQ_SIZE = 3
PADDING = False
VOCAB_SIZE = 278 + 2 if (PADDING) else 278 # to account for padding
# model params
LSTM_UNITS = 64
EPOCHS = 4
BATCH_SIZE = 128
DATASET_SIZE = 1000 * 1000
VALIDATION_SPLIT = 0.2
ENCODER = one_hot_encode

# helper to print config
def configToString():
    return f'{ENCODER.__name__}-{DATASET_SIZE}-{BATCH_SIZE}-{LSTM_UNITS}-{EPOCHS}-{PADDING}'

print('Running config:', configToString())
print('Loading data...')
x, word_x = load_data(num_scripts=DATASET_SIZE, padding=PADDING)
x_word = {v: k for k, v in word_x.items()}
print('Encoding the words..')
one_hot_x = encode_dataset(x, one_hot_encode)
print('Building the y labels..')
xs, ys = build_xy(one_hot_x)
# make it a numpy array
xs = np.array(xs, dtype=np.bool)
ys = np.array(ys, dtype=np.bool)

# print('Sample training sequence', x_train[0])
print(len(xs)*(1-VALIDATION_SPLIT), 'train sequences')
print(len(xs)*VALIDATION_SPLIT, 'test sequences')
print('X shape', np.shape(xs))
print('Y shape', np.shape(ys))

print('Building the model')
model = Sequential()
model.add(LSTM(LSTM_UNITS, input_shape=(SEQ_SIZE, VOCAB_SIZE)))
model.add(Dense(VOCAB_SIZE))
model.add(Activation('softmax'))

print('Training..')
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(xs, ys, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True).history

print('saving the model and history..')
model.save( RESULTS_DIR + f'/rnn-{configToString()}.h5')
pickle.dump(history, open( RESULTS_DIR + f"/history-{configToString()}.p", "wb"))

# load em back
model = load_model( RESULTS_DIR + f'/rnn-{configToString()}.h5')
history = pickle.load(open( RESULTS_DIR + f"/history-{configToString()}.p", "rb"))

#plot 
print('plotting..')
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title(f'Model Accuracy. {configToString()}')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left');
plt.savefig(RESULTS_DIR + f'/acc-{configToString()}.png', bbox_inches='tight')

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title(f'Model Loss. {configToString()}')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left');
plt.savefig(RESULTS_DIR + f'/loss-{configToString()}.png', bbox_inches='tight')