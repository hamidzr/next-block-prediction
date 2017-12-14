from keras.models import Sequential, load_model
import heapq
import numpy as np
import math
import sys
import pickle
from utils.helpers import load_wv, script_tokenizer

MODEl_PATH = './results/rnn-embedding_encode-mse-500000-128-128-10-3-False.h5'
# TODO set a global config file
SEQ_SIZE = 4


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

# return the top n probable prediction
def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def next_block(blockSeq):
    input_vecs = prepare_input(blockSeq)
    print(input_vecs.shape)
    preds = model.predict(input_vecs, verbose=1)
    print(preds.shape)
    print('preds', preds)
    next_vec = preds[0]
    next_block = wv.similar_by_vector(next_vec, topn=1, restrict_vocab=None)
    return next_block, next_vec

# TODO autogenerate scripts

# def predict_completion(text):
#     original_text = text
#     generated = text
#     completion = ''
#     while True:
#         x = prepare_input(text)
#         preds = model.predict(x, verbose=0)[0]
#         next_vec = sample(preds, top_n=1)[0]
#         next_block = wv.similar_by_vector(next_vec, topn=1, restrict_vocat=None)

#         text = text[1:] + next_char
#         completion += next_char

#         if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
#             return completion


# def predict_completions(text, n=3):
#     x = prepare_input(text)
#     preds = model.predict(x, verbose=0)[0]
#     next_indices = sample(preds, n)
#     return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]

while True:
    seqText = input('type a valid seq: ')
    blockSeq = script_tokenizer(seqText)
    print(next_block(blockSeq))
    # vec = wv['whenKeyPressed']
    # similar = wv.similar_by_vector(vec, topn=1)
    # print(similar)
