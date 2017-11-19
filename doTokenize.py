#!/bin/python3.6
import nltk
import sys
from tqdm import tqdm
import pickle
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
import argparse
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--textFile", help="text document")
parser.add_argument("pickleFile", help="tokens file location")
args = parser.parse_args()
num_cores = multiprocessing.cpu_count()


sentenceTokens = []
# load or calculate the tokens
if (args.textFile == None):
    # load from pickle
    print('loading from ', args.pickleFile)
    with open(args.pickleFile, 'rb') as f:
        tokens = pickle.load(f)
else:
    with open(args.textFile, 'r') as myfile:
        sentenceTokens = Parallel(n_jobs=num_cores)(delayed(nltk.word_tokenize)(line) for line in tqdm(myfile.readlines()))

    # concat sentenceTokens into one list
    tokens = []
    for sentence in sentenceTokens:
        tokens += sentence;
    print('finished tokenizing', len(tokens), 'tokens')
    with open(args.pickleFile, 'wb') as f:
        pickle.dump(tokens, f)

