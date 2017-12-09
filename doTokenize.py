#!/bin/python3.6
import nltk
import math
import sys
from tqdm import tqdm
import pickle
from nltk.util import ngrams
from collections import Counter
import argparse
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from utils.helpers import script_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--textFile", help="text document")
parser.add_argument("--blocksPickleFile", help="1gram counts tuples output path")
parser.add_argument("pickleFile", help="tokens file location")
args = parser.parse_args()
num_cores = multiprocessing.cpu_count()


# calcuate frequency stats
def calcStats(tokenStats):
    lowFreqs, highFreqs, unused = clusterTokens(tokenStats)
    print('# Unique blocks', len(tokenStats.items()))
    print('# Low freq blocks', len(lowFreqs))
    print('# sensible blocks', len(highFreqs))


def clusterTokens(tokenStats):
    LOW_THRESHOLD = 10 # blocks with lesser counts will be considered low freq
    lowFreqs = list(filter(lambda pair: pair[1] < LOW_THRESHOLD, tokenStats.items()))
    highFreqs = list(filter(lambda pair: pair[1] >= LOW_THRESHOLD, tokenStats.items()))
    langBlocks = []
    for p in highFreqs:
        if p[0] not in langBlocks:
            langBlocks.append(p[0])
    return lowFreqs, highFreqs, langBlocks

sentenceTokens = []
# load or calculate the tokens
if (args.textFile == None):
    # load from pickle
    print('loading from ', args.pickleFile)
    with open(args.pickleFile, 'rb') as f:
        tokens = pickle.load(f)
else:
    with open(args.textFile, 'r') as myfile:
        sentenceTokens = Parallel(n_jobs=num_cores)(delayed(script_tokenizer)(line) for line in tqdm(myfile.readlines()))

    # concat sentenceTokens into one list
    tokens = []
    for sentence in sentenceTokens:
        tokens += sentence;
    print('finished tokenizing', len(tokens), 'tokens')
    tokenStats = Counter(tokens)
    calcStats(tokenStats)
    lowF, highF, lang = clusterTokens(tokenStats)
    with open(args.pickleFile, 'wb') as f:
        tokens = list(filter(lambda t: t in lang, tokens))
        print(f'writing {len(tokens)} to file')
        pickle.dump(tokens, f)

def plotFrequencyDist(tokenCounter, num_blocks=50, scaled=False):
    labels, counts = list(map(list, zip(*tokenCounter.most_common(num_blocks))))

    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(labels)]

    plt.xlabel("Block")
    plt.ylabel("Count")
    plt.title("Block Frequency")
    plt.xticks(x_pos, labels)
    plt.xticks(rotation='vertical')

    if scaled:
        counts = list(map(lambda x: math.log(x,2), counts)) # log base count
        plt.ylabel("Log-count")

    plt.bar(x_pos, counts, color='green')
    plt.show()

tokenStats = Counter(tokens)
with open(args.blocksPickleFile, 'wb') as f:
    pickle.dump(list(tokenStats.items()), f)
plotFrequencyDist(tokenStats)

bigrams = ngrams(tokens, 2)
bigramStats = Counter(bigrams)
print('# bigram unique blocks', len(bigramStats.items()))
plotFrequencyDist(bigramStats)

trigrams = ngrams(tokens, 3)
trigramStats = Counter(trigrams)
print('# trigram unique blocks', len(trigramStats.items()))
plotFrequencyDist(trigramStats)
