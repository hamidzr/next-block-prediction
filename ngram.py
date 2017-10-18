#!/bin/python3.6
import nltk
import numpy as np
import sys
import pickle
import time
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

# get tokens
text = "I need to write a program in NLTK that breaks a corpus (a large collection of txt files) into unigrams, bigrams, trigrams, fourgrams and fivegrams. I need to write a program in NLTK that breaks a corpus"

counter = 1
tokens = []

if (len(sys.argv) > 1):
    tokensPath = sys.argv[1];
    with open(tokensPath, 'rb') as f:
        tokens = pickle.load(f)

print('loaded', len(tokens), 'tokens')

# # create n-grams
# bigrams = ngrams(tokens,2)
# # trigrams = ngrams(tokens,3)
# # fourgrams = ngrams(tokens,4)
# # fivegrams = ngrams(tokens,5)

# gramStats = Counter(bigrams)
# print(gramStats.most_common(500))

# while True:
#     word = input('input words ')
#     results = []
#     totalCount = 0
#     for gram, count in gramStats.items():
#         if (gram[0] == word):
#             totalCount += count
#             results.append([gram[1], count])
#     results.sort(key=lambda x: x[1])
#     for block, count in results:
#         print(block, round(count/totalCount, 3))

trigrams = ngrams(tokens,3)
# fourgrams = ngrams(tokens,4)
# fivegrams = ngrams(tokens,5)

gramStats = Counter(trigrams)
print(gramStats.most_common(200)

while True:
    word = input('input words ')
    results = []
    totalCount = 0
    for gram, count in gramStats.items():
        if (' '.join([gram[0], gram[1]]) == word):
            totalCount += count
            results.append([gram[2], count])
    results.sort(key=lambda x: x[1])
    for block, count in results[0:10]:
        print(block, round(count/totalCount, 3), count)

# TODO calculate the sparsity
