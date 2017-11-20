#!/bin/python3.6
import nltk
import math
import numpy as np
import sys
import pickle
import time
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("-n", "--ngrams", help="ngram parameter", type=int)
parser.add_argument("--tokens", help="tokens pickle file address")
args = parser.parse_args()

if args.verbose:
    print("verbosity turned on")

tokens = []

if (args.tokens):
    with open(args.tokens, 'rb') as f:
        tokens = pickle.load(f)
else:
    print('tokens are missing, use the provided tokenizer.')

print('loaded', len(tokens), 'tokens')

print('calculating', args.ngrams, 'grams')
gramed = ngrams(tokens,args.ngrams)

# TODO IMP add starting and ending WORDs ?
# NOTE alan intori tahe jomle ghabli chasbide be badi
gramStats = Counter(gramed)
print(gramStats.most_common(50))

# memoize helper
def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper

# TODO add kney and add-1 smoothing
# idea sequence is ngramSize-1
# returns a dictionary of probabilities
def simpleProbabilities(sequence):
    windowSize = args.ngrams-1
    if len(sequence) <  windowSize: raise Exception('short sequence') #TODO backoff to lower grams? 
    results = []
    probabilities = {}
    totalCount = 0
    for gram, count in gramStats.items():
        if (list(gram[0:windowSize]) == sequence[-windowSize:]):
            totalCount += count
            results.append([gram[-1:][0], count])
    results.sort(key=lambda x: x[1])
    for candidateBlock, count in results:
        probab = round(count/totalCount, 10)
        probabilities[candidateBlock] = probab
    # or return a sorted list of (block, prob) pairs  
    return probabilities

# calculates the perplexity for a sequence of blocks
def perplexity(blockSequence):
    windowSize = args.ngrams -1
    numWords = len(blockSequence)
    sequenceProbabilityInv = 1;
    # calculate sequence prob inv
    for idx, val in enumerate(blockSequence):
        if idx < windowSize: continue # skip the first n blocks. change if you added starting padding
        probs = simpleProbabilities(blockSequence[idx-windowSize:idx])
        invProb = 1.0/probs[val]
        sequenceProbabilityInv = sequenceProbabilityInv * invProb
    perplexity = (sequenceProbabilityInv)**(1.0/ numWords)
    print(perplexity)
    return perplexity

while True:
    # calculate probabilities given a sequence of words
    seq = input('pass a sequence: ')
    seq = seq.split(' ')
    # simpleProbabilities(seq, gramStats)
    perplexity(seq, gramStats)

