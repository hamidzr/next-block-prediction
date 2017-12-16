from keras.models import Sequential, load_model
import heapq
import numpy as np
import math
import sys
import pickle
from utils.helpers import load_wv, script_tokenizer

MODEl_PATH = './results/rnn-embedding_encode-mse-9999000-128-128-8-5-3-False.h5'
# TODO set a global config file
SEQ_SIZE = 5


# wv model must be the same as the one used to train the model (dimensions etc)
wv = load_wv(False)
model = load_model(MODEl_PATH)


# in array of blocks
# WARN cuts down to last seqSize
def prepare_input(blockSeq):
    # get last seqSize blocks
    modelInput = []
    inpBlocks = blockSeq[-SEQ_SIZE:]
    for block in inpBlocks:
        try:
            vec = wv[block]
            modelInput.append(vec)
        except Exception as e:
            raise e
            # raise 'invalid input block'
    return np.array([modelInput])

def next_block(blockSeq):
    input_vecs = prepare_input(blockSeq)
    preds = model.predict(input_vecs, verbose=1)
    next_vec = preds[0]
    next_blocks = wv.similar_by_vector(next_vec, topn=3, restrict_vocab=None)
    return next_blocks, next_vec

# TODO autogenerate scripts

while True:
    seqText = input('type a valid seq: ')
    blockSeq = script_tokenizer(seqText)
    print(next_block(blockSeq)[0])
