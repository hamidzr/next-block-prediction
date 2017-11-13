#!/bin/python3.6
import nltk
import sys
from tqdm import tqdm
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

with open(args.textFile, 'r') as myfile:
    for line in tqdm(myfile.readlines()):
        tokens += nltk.word_tokenize(line)

print('finished tokenizing', len(tokens), 'tokens')
with open(args.output, 'wb') as f:
    pickle.dump(tokens, f)

