#!/bin/python3.6
import nltk
import math
import numpy as np
import sys
from tqdm import tqdm
import pickle
import time
from nltk.util import ngrams
from collections import Counter
import argparse
from joblib import Parallel, delayed
import multiprocessing
from utils.helpers import script_tokenizer, memoize
num_cores = multiprocessing.cpu_count()

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("-n", "--ngrams", help="ngram parameter", type=int)
parser.add_argument("--tokens", help="tokens pickle file address")
args = parser.parse_args()

TEST_SET = './data/scratch/scripts_sample.txt'

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


# ideal sequence is ngramSize-1
# compute for a single word
def noSmoothing(sequence, block):
    windowSize = args.ngrams-1
    if len(sequence) !=  windowSize: raise Exception(f'mismatching sequence. {len(sequence)}')
    totalCount = countGramsStartingWith(sequence)
    blockCount = ngramsCount(sequence, block)
    p = blockCount/float(totalCount)
    return p

# TODO normalize the counts
def addOneSmoothing(sequence, block):
    windowSize = args.ngrams-1
    if len(sequence) !=  windowSize: raise Exception(f'mismatching sequence. {len(sequence)}')
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
def lambdaWeight(sequence, block):
    # H mashkook makhrajo motmaen nisti
    windowSize = args.ngrams-1
    if len(sequence) !=  windowSize: raise Exception('bad sequence length')
    uniqueCount = 0
    for gram, count in gramStats.items():
        if (list(gram)[0:windowSize] == sequence):
            uniqueCount += 1
    return uniqueCount/blockStats[block]


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
    prob = discountedGram + lambdaWeight(sequence, block)*continuationProbability(block)
    return prob

####### end of Kneser-Ney ##########

# calculates the perplexity for a sequence of blocks
def perplexity(blockSequence):
    windowSize = args.ngrams -1
    numWords = len(blockSequence)
    if (numWords < windowSize ):
        print('invalid short input sequence')
        return -1
    sequenceProbabilityInv = 1;
    # calculate sequence prob inv
    for idx, val in enumerate(blockSequence):
        if idx < windowSize: continue # skip the first n blocks. change if you added starting padding
        prob = probabilityFn(blockSequence[idx-windowSize:idx], val)
        # print(prob)
        if (prob > 1): raise Exception('calculated a probability more than 1!')
        invProb = 1.0/prob
        sequenceProbabilityInv = sequenceProbabilityInv * invProb
    perplexity = (sequenceProbabilityInv)**(1.0/ numWords)
    return perplexity

def evaluate(testSet):
    # tokenize each sentence of test set
    with open(testSet, 'r') as f:
        sentenceTokens = Parallel(n_jobs=num_cores)(delayed(script_tokenizer)(line) for line in tqdm(f.readlines()))
    # calculate perplexity for each sent
    print('calculating perplexities')
    perps = Parallel(n_jobs=num_cores)(delayed(perplexity)(sent) for sent in tqdm(sentenceTokens))
    perps = list(filter(lambda p: p > 0, perps))
    avg = sum(perps) / float(len(perps))
    print(avg)
    return avg

def interactiveInspection():
    while True:
        # calculate probabilities given a sequence of words
        seq = input('pass a sequence: ')
        seq = seq.split(' ')
        preds = []
        for block in blockStats:
            prob = probabilityFn(seq[-args.ngrams+1:], block)
            preds.append((block, prob))
        preds.sort(key=lambda tup: tup[1], reverse=True)
        print('predictions:', preds[:3])
        print('perplexity:', perplexity(seq))



######### driver ##########

# load block tokens
tokens = loadTokens()

# calculates ngrams and stats
print('calculating', args.ngrams, 'grams')
gramed = ngrams(tokens,args.ngrams)
# TODO IMP add starting and ending WORDs ?
# NOTE alan intori tahe jomle ghabli chasbide be badi
gramStats = Counter(gramed)
print('# unique tokens', len(gramStats.items()))
blockStats = Counter(tokens)
print(gramStats.most_common(100))

# set the desired probabilityFn
probabilityFn = addOneSmoothing

interactiveInspection()
#evaluate(TEST_SET)
