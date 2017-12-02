# cleans the dataset
import sys
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input file")
parser.add_argument("--output", help="output file")
args = parser.parse_args()

MAX_SCRIPT_SIZE = os.getenv(MAX_SCRIPT_SIZE, 40)
sentenceSizes = []

def removeSpecialChars(line):
    # line.replace(':','')
    # line.replace('. .','')
    return line

# true if it has a reasonable size
def isBadSized(line):
    length = len(line.split(' '))
    sentenceSizes.append(length)
    if (length < 2 or length > MAX_SCRIPT_SIZE):
        print('removed script of size', length)
        print(line)
        return True
    return False

with open(args.input, 'r') as inFile:
    with open(args.output, 'w') as outFile:
        for line in inFile.readlines():
            # apply a series of cleanup then writeout or not
            line = removeSpecialChars(line)
            if (isBadSized(line)): continue
            outFile.write(line)



def plotSizeDist():
    ys = sentenceSizes
    plt.ylabel("Count")
    plt.xlabel("Length")
    plt.title("Script Length Histogram")
    plt.hist(ys, range=(1, MAX_SCRIPT_SIZE), facecolor='b')
    plt.show()

print('avg script size', sum(sentenceSizes)/len(sentenceSizes))
plotSizeDist()
