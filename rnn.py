import numpy as np
import math
np.random.seed(42)
import itertools
import tensorflow as tf
tf.set_random_seed(42)
from sklearn.preprocessing import scale
from keras.models import Sequential, load_model
from keras.preprocessing.text import one_hot, text_to_word_sequence
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout, GRU
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
from gensim.models.word2vec import Word2Vec
from pylab import rcParams
from joblib import Parallel, delayed
from utils import constants
from utils.helpers import load_wv, memoize

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 12, 5

# pad the scripts with begining and ending
def pad_script(script):
    # add seq_size -1 to the begining of the script
    script.append(constants.END_SYMBOL)
    for _ in range(0, SEQ_SIZE-1):
        script.insert(0, constants.START_SYMBOL)
    return script

def load_data(file_path, num_scripts=100000, padding=False):
    print(f'loading {num_scripts} lines of data..')
    tokens_list = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            line_words = text_to_word_sequence(line, filters='\n', lower=False)
            # line_words = helpers.script_tokenizer(line)
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

blockVectors = False
@memoize
def embedding_encode(number):
    # number to block
    block = x_word[number]
    return blockVectors[block]

def encode_dataset(x, encoder):
    encodedSet = []
    failedCounter = 0
    for seq in x:
        encoded = []
        for blockNumber in seq:
            try:
                encBlock = encoder(blockNumber)
                encoded.append(encBlock)
            except Exception as e:
                print(e)
                failedCounter += 1
        encodedSet.append(encoded)
    print(f'failed to encode {failedCounter} block instances')
    return encodedSet


# split a list randomly into two with the given ratio
def split_data(aList, ratio=0.8):
    np.random.shuffle(aList)
    splitPoint = int(ratio * len(aList))
    return aList[0:splitPoint], aList[splitPoint:]

# build lists of sequences and the actual next word for that response
# builds out the sequences and the corresponding next word
def build_xy(scripts_list):
    sequences = []
    next_words = []
    for script in scripts_list:
        scriptSize = len(script)
        for idx in range(0, scriptSize-SEQ_SIZE):
            sequences.append(script[idx:idx+SEQ_SIZE])
            next_words.append(script[idx+SEQ_SIZE])
    return sequences, next_words

def plot():
    #plot 
    print('plotting..')
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title(f'Model Accuracy. {configToString()}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left');
    plt.savefig(constants.RESULTS_DIR + f'/acc-{configToString()}.png', bbox_inches='tight')
    plt.close() # clear fig

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(f'Model Loss. {configToString()}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left');
    plt.savefig(constants.RESULTS_DIR + f'/loss-{configToString()}.png', bbox_inches='tight')

# helper to print config
def configToString():
    return f'{ENCODER.__name__}-{LOSS}-{DATASET_SIZE}-{BATCH_SIZE}-{LSTM_UNITS}-{BLOCK_VEC_SIZE}-{SEQ_SIZE}-{EPOCHS}-{PADDING}'

#### parameters

SEQ_SIZE = 4
PADDING = False
VOCAB_SIZE = 'unknown' # will be set after loading the dataset
# model params
LSTM_UNITS = 128
EPOCHS = 3
BATCH_SIZE = 128
DATASET_SIZE = 500 * 1000
VALIDATION_SPLIT = 0.1
ENCODER = embedding_encode
OPTIMIZER = RMSprop(lr=0.01) # 'adam'
DROPOUT = 0.01
RNN_CELL = GRU

x, word_x = load_data(constants.INPUT, num_scripts=DATASET_SIZE, padding=PADDING)
VOCAB_SIZE = len(word_x) + 2 if (PADDING) else len(word_x) # calc vocab size

## auto set
if (ENCODER == embedding_encode):
    blockVectors = load_wv(False)
    BLOCK_VEC_SIZE = constants.WORD2VEC_SIZE # if word2vec == vec size
    LOSS = 'mse'
else:
    BLOCK_VEC_SIZE = VOCAB_SIZE
    LOSS = 'categorical_crossentropy'

print('Running config:', configToString())
x_word = {v: k for k, v in word_x.items()}
print('Encoding the words..')
encoded_x = encode_dataset(x, ENCODER)
print('Building the y labels..')
xs, ys = build_xy(encoded_x)
# make it a numpy array
xs = np.array(xs)
ys = np.array(ys)

print('Sample training sequence', xs[0])
print(math.floor(len(xs)*(1-VALIDATION_SPLIT)), 'train sequences')
print(math.floor(len(xs)*VALIDATION_SPLIT), 'test sequences')
print('X shape', np.shape(xs))
print('Y shape', np.shape(ys))

print('Building the model')
model = Sequential()
# model.add(Embedding(VOCAB_SIZE, 100, input_length=10))
model.add(RNN_CELL(LSTM_UNITS, dropout=DROPOUT, recurrent_dropout=0, input_shape=(SEQ_SIZE, BLOCK_VEC_SIZE)))
model.add(Dense(BLOCK_VEC_SIZE))
model.add(Activation('softmax'))

print('Training..')
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(xs, ys, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True).history
print(model.summary())

print('saving the model and history..')
model.save( constants.RESULTS_DIR + f'/rnn-{configToString()}.h5')
pickle.dump(history, open( constants.RESULTS_DIR + f"/history-{configToString()}.p", "wb"))

# load em back
model = load_model( constants.RESULTS_DIR + f'/rnn-{configToString()}.h5')
history = pickle.load(open( constants.RESULTS_DIR + f"/history-{configToString()}.p", "rb"))

plot()

