# cleans the dataset
# input raw output of snapAnalyzer (raw project texts)
import sys
import re
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input file")
parser.add_argument("--output", help="output file")
args = parser.parse_args()

MAX_SCRIPT_SIZE = os.getenv('MAX_SCRIPT_SIZE', 40)
SCRIPT_SYMBOL = 'ESC'
SPRITE_SYMBOL = 'ESP'
scriptSizes = []
spriteCounts = [] # one entry per project

def removeSpecialChars(line):
    # line.replace(':','')
    line.replace('\n','')
    line.replace('. .','')
    return line

# true if it has a reasonable size
def isBadSized(line):
    length = len(line.split(' '))
    scriptSizes.append(length)
    if (length < 2 or length > MAX_SCRIPT_SIZE):
        # print('removed script of size', length)
        # print(line)
        return True
    return False

def removeMultiSpaces(line):
    return re.sub(r"", " ", line)

# convert to 1d list of scripts
def projectToScripts(projectLine):
    sprites = projectLine.split(SPRITE_SYMBOL)
    spriteCounts.append(len(sprites))
    scripts = []
    for sprite in sprites:
        scripts += projectLine.split(SCRIPT_SYMBOL)
    return scripts


with open(args.input, 'r') as inFile:
    with open(args.output, 'w') as outFile:
        # for project in inFile.readlines(): #read line by line for bigger datasets
        fileLines = inFile.read().split('\n')
        for project in tqdm(fileLines):
            for line in projectToScripts(project):
                # apply a series of cleanup then writeout or not
                line = removeSpecialChars(line)
                line = removeMultiSpaces(line)
                if (isBadSized(line)): continue
                outFile.write(line)


def plotSizeDist():
    ys = scriptSizes
    plt.ylabel("Count")
    plt.xlabel("Length")
    plt.title("Script Length Histogram")
    plt.hist(ys, range=(1, MAX_SCRIPT_SIZE), facecolor='b')
    plt.show()

print('avg script size', sum(scriptSizes)/len(scriptSizes))
plotSizeDist()
