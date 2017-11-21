#!/bin/python3.6
import nltk
import math
import numpy as np
import sys
from tqdm import tqdm
import pickle
import time
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
import argparse
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("-n", "--ngrams", help="ngram parameter", type=int)
parser.add_argument("--tokens", help="tokens pickle file address")
args = parser.parse_args()

if args.verbose:
    print("verbosity turned on")

def loadTokens():
    tokens = []
    if (args.tokens):
        with open(args.tokens, 'rb') as f:
            tokens = pickle.load(f)
    else:
        print('tokens are missing, use the provided tokenizer.')

    print('loaded', len(tokens), 'tokens')
    return tokens


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

# helper count grams
@memoize
def countGramsStartingWith(sequence):
    windowSize = len(sequence)
    if windowSize >= args.ngrams : raise Exception('bad sequence length')
    totalCount = 0
    for gram, count in gramStats.items():
        if (list(gram[0:windowSize]) == sequence[0:windowSize]):
            totalCount += count
    return totalCount

# helper looksup ngram count of a seq + last block
# how many times this block came after seq
@memoize
def ngramsCount(seq, block):
    targetSeq = tuple(seq) + (block,)
    for gram, count in gramStats.items():
        if (gram == targetSeq): return count
    return 0


# idea sequence is ngramSize-1
# compute for a single word
def noSmoothing(sequence, block):
    windowSize = args.ngrams-1
    if len(sequence) !=  windowSize: raise Exception('short sequence') #TODO backoff to lower grams? 
    totalCount = countGramsStartingWith(sequence)
    blockCount = ngramsCount(sequence, block)
    p = blockCount/float(totalCount)
    return p

# TODO normalize the counts
def addOneSmoothing(sequence, block):
    windowSize = args.ngrams-1
    if len(sequence) !=  windowSize: raise Exception('short sequence') #TODO backoff to lower grams? 
    totalCount = countGramsStartingWith(sequence)
    blockCount = ngramsCount(sequence, block)
    if (blockCount == 0): blockCount += 1
    p = blockCount/float(totalCount)
    return p



######### Kneser-Ney and Absolute Discounting ##########
# how likely is a block to appear as a novel continuation
@memoize
def continuationProbability(block):
    # count how many different types block B compelets
    novelCounts = 0
    for gram, count in gramStats.items():
        if (list(gram)[-1:][0] == block):
            novelCounts += 1
    return novelCounts/float(len(gramStats.items()))

# number of single block types that can follow a sequence
@memoize
def lambdaWeight(sequence):
    windowSize = args.ngrams-1
    if len(sequence) !=  windowSize: raise Exception('bad sequence length')
    uniqueCount = 0
    for gram, count in gramStats.items():
        if (list(gram)[0:windowSize] == sequence):
            uniqueCount += 1
    return uniqueCount


# sequence is one smaller in length from ngram
# given sequence what is the probability of block following
# Kneser-Ney smoothing
def KNSmoothing(sequence, block):
    # absolute discounting value (different for low counts 1,2)
    windowSize = args.ngrams-1
    if (len(sequence) !=  windowSize): raise Exception('bad sequence length')
    totalCount = countGramsStartingWith(sequence)
    blockCount = ngramsCount(sequence, block) # how many times this block was next in seq
    d = 0.75
    if (blockCount < 2): d = 0.4
    discountedGram = (blockCount - d)/float(totalCount)
    prob = discountedGram + lambdaWeight(sequence)*continuationProbability(block)
    return prob

####### end of Kneser-Ney ##########

# calculates the perplexity for a sequence of blocks
def perplexity(blockSequence):
    windowSize = args.ngrams -1
    numWords = len(blockSequence)
    sequenceProbabilityInv = 1;
    # calculate sequence prob inv
    for idx, val in enumerate(blockSequence):
        if idx < windowSize: continue # skip the first n blocks. change if you added starting padding
        prob = probabilityFn(blockSequence[idx-windowSize:idx], val)
        invProb = 1.0/prob
        sequenceProbabilityInv = sequenceProbabilityInv * invProb
    perplexity = (sequenceProbabilityInv)**(1.0/ numWords)
    return perplexity

def evaluate(testSet):
    # tokenize each sentence of test set
    with open(testSet, 'r') as f:
        sentenceTokens = Parallel(n_jobs=num_cores)(delayed(nltk.word_tokenize)(line) for line in tqdm(f.readlines()))
    # calculate perplexity for each sent
    print('calculating perplexities')
    perps = Parallel(n_jobs=1)(delayed(perplexity)(sent) for sent in tqdm(sentenceTokens))
    avg = sum(perps) / float(len(perps))
    print(avg)
    return avg

def interactiveInspection():
    while True:
        # calculate probabilities given a sequence of words
        seq = input('pass a sequence: ')
        seq = seq.split(' ')
        # probabilityFn(seq, gramStats)
        perplexity(seq)



######### driver ##########

# load block tokens
tokens = loadTokens()

# calculates ngrams and stats
print('calculating', args.ngrams, 'grams')
gramed = ngrams(tokens,args.ngrams)
# TODO IMP add starting and ending WORDs ?
# NOTE alan intori tahe jomle ghabli chasbide be badi
gramStats = Counter(gramed)
blockStats = Counter(tokens)
print(gramStats.most_common(10))

# set the desired probabilityFn
probabilityFn = noSmoothing

evaluate('data/sample.txt')
