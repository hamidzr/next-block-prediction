import numpy as np
np.random.seed(42)
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

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 12, 5

INPUT = './data/cleanSample.txt'
START_SYMBOL = 'START'
END_SYMBOL = 'END'

# def load_data_flat(num_scripts=100000):
#     tokens_list = []
#     text = ''
#     with open(INPUT, 'r') as f:
#         for idx, line in enumerate(f.readlines()):
#             text += line + NEWLINE_SYMBOL
#             if (idx >= num_scripts): break
#     # vocab_size = len(set(text_to_word_sequence(text)))
#     oneHotVectors = one_hot(text, vocab_size)
#     return oneHotVectors

# pad the scripts with begining and ending
def pad_script(script):
    # add seq_size -1 to the begining of the script
    script.append(END_SYMBOL)
    for _ in xrange(SEQ_SIZE-1):
        script.insert(0, START_SYMBOL)
    return script

def load_data(num_scripts=100000):
    tokens_list = []
    with open(INPUT, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            line_words = text_to_word_sequence(line)
            line_words = pad_script(line_words)
            tokens_list.append(line_words)
            if (idx >= num_scripts): break
    all_tokens = itertools.chain.from_iterable(tokens_list)
    word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
    # convert token lists to token-id lists, e.g. [[1, 2], [2, 2]] here
    token_ids = [[word_to_id[token] for token in tokens_doc] for tokens_doc in tokens_list]
    return token_ids, word_to_id

def split_data(aList, ratio=0.8):
    np.random.shuffle(aList)
    splitPoint = int(ratio * len(aList))
    return aList[0:splitPoint], aList[splitPoint:]

def build_xy():
    pass

VOCAB_SIZE = 280 #real vocab size + 2 for padding
SEQ_SIZE = 3

print('Loading data...')
x, word_x = load_data(num_scripts=1000)
x_train, x_test = split_data(x, 0.9)
print('Sample of training first 5 training sequence', x_train[0:5])
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))
