#!/bin/python3.6
import nltk
import sys
import pickle
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter


tokens = []
counter = 1

if (len(sys.argv) > 2):
    textPath = sys.argv[1];
    outPath = sys.argb[2];
    # with open(textPath, 'r') as myfile:
    #     text=myfile.read()

    with open(textPath, 'r') as myfile:
        for line in myfile.readlines():
            tokens += nltk.word_tokenize(line)
            if (counter % 1000) == 0: print(counter)
            counter +=1
else:
    tokens += nltk.word_tokenize(text)

print('finished tokenizing', len(tokens), 'tokens')
with open(outPath, 'wb') as f:
    pickle.dump(tokens, f)

