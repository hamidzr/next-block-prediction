#!/bin/python3.6
import nltk
import sys
import pickle
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("textFile", help="text document")
parser.add_argument("output", help="generated dict of tokens as a pickle file")
args = parser.parse_args()

tokens = []
counter = 1

with open(args.textFile, 'r') as myfile:
    for line in myfile.readlines():
        tokens += nltk.word_tokenize(line)
        if (counter % 1000) == 0: print(counter)
        counter +=1

print('finished tokenizing', len(tokens), 'tokens')
with open(args.output, 'wb') as f:
    pickle.dump(tokens, f)

