# cleans the dataset
# input raw output of snapAnalyzer (raw project texts)
import sys
import pickle
import re
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from utils.helpers import get_num_lines
num_cores = multiprocessing.cpu_count()

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input file")
parser.add_argument("--output", help="output file")
args = parser.parse_args()

MAX_SCRIPT_SIZE = os.getenv('MAX_SCRIPT_SIZE', 20)
MIN_SCRIPT_SIZE = os.getenv('MIN_SCRIPT_SIZE', 4)
MAX_SPRITE_COUNT = 10
SCRIPT_SYMBOL = ' ESC '
SPRITE_SYMBOL = ' ESP '
scriptSizes = []
spriteCounts = [] # one entry per project

# remove duplicate items in list
def removeDuplicates(inp):
    unique = []
    for i in inp:
      if i not in unique:
        unique.append(i)
    return unique

def removeSpecialChars(line):
    # line.replace(':','')
    line.replace('\n','')
    line.replace('. .','')
    return line

# true if it has a reasonable size
badScripts = 0;
def isBadSized(line):
    global badScripts
    length = len(line.split(' '))
    scriptSizes.append(length)
    if (length < MIN_SCRIPT_SIZE or length > MAX_SCRIPT_SIZE):
        # print('removed script of size', length)
        # print(line)
        badScripts += 1
        return True
    return False

def removeMultiSpaces(line):
    return re.sub(r"\ {2,}", " ", line)

# convert to 1d list of scripts
badProjects = 0;
def projectToScripts(projectLine):
    global badProjects
    sprites = projectLine.split(SPRITE_SYMBOL)
    spriteCounts.append(len(sprites))
    sprites = removeDuplicates(sprites)
    if (len(sprites) > MAX_SPRITE_COUNT):
        badProjects += 1
        return []

    scripts = []
    for sprite in sprites:
        ss = sprite.split(SCRIPT_SYMBOL)
        ss = list(filter(lambda s: not isBadSized(s), ss))
        ss = removeDuplicates(ss)
        scripts += ss
    return scripts


# to better script
def transformAndClean(line):
    # apply a series of cleanup then writeout or not
    line = removeSpecialChars(line)
    line = removeMultiSpaces(line)
    return line

print('loading input..')
inputLines = ''
with open(args.input, 'r') as inFile:
    # for project in inFile.readlines(): #read line by line for bigger datasets
    inputLines = inFile.read().split('\n')
print('breaking projects to scripts.. ')
lines = []
for projectLine in tqdm(inputLines):
    for scriptLine in projectToScripts(projectLine):
        lines.append(scriptLine)
print('cleaning the scripts..')
scripts = Parallel(n_jobs=num_cores)(delayed(transformAndClean)(line) for line in tqdm(lines))
print(f'removed {badProjects} projects')
print(f'removed {badScripts} scripts')
print('writing out..')
with open(args.output, 'w') as outFile:
    for s in scripts:
        outFile.write(f'{s}\n')

with open(args.output + '.pickle', 'wb') as f:
    stats  = {'scriptSizes': scriptSizes, 'spriteCounts': spriteCounts}
    pickle.dump(stats, f)

def plotSizeDist():
    ys = scriptSizes
    plt.ylabel("Prevelance")
    plt.xlabel("Length")
    plt.title("Script Length Histogram")
    plt.hist(ys, range=(MIN_SCRIPT_SIZE, MAX_SCRIPT_SIZE), facecolor='b')
    plt.show()

def plotSpriteCount():
    ys = spriteCounts
    plt.ylabel("Prevelance")
    plt.xlabel("Count")
    plt.title("Sprite Count Histogram")
    plt.hist(ys, range=(1, MAX_SPRITE_COUNT), facecolor='g')
    plt.show()

scriptSizes = list(filter(lambda s: s > MIN_SCRIPT_SIZE and s < MAX_SCRIPT_SIZE, scriptSizes))
spriteCounts = list(filter(lambda s: s < MAX_SPRITE_COUNT, spriteCounts))
print('avg script size', sum(scriptSizes)/len(scriptSizes))
print('avg sprite count', sum(spriteCounts)/len(spriteCounts))
plotSizeDist()
plotSpriteCount()
