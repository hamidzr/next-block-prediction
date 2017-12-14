from functools import wraps
from gensim.models.word2vec import Word2Vec
import pickle
from .constants import *
from sklearn.preprocessing import scale
import mmap

lang = []
try:
    with open(BLOCK_LANG, 'rb') as f:
        lang = pickle.load(f)
except Exception as e:
    print(e)
    print('language file unavailable', BLOCK_LANG)

# memoize helper
def memoize(f):
    cache = {}
    @wraps(f)
    def decorated(*args):
        key = (f, str(args))
        result = cache.get(key, None)
        if result is None:
            result = f(*args)
            cache[key] = result
        return result
    return decorated

# takes in the script as a one liner string
def script_tokenizer(script):
    # keep all the punctuation
    tokens = script.split(' ')
    tokens[-1] = tokens[-1].replace('\n', '')
    # get rid of unknown tokens
    if (len(lang) > 10): tokens = list(filter(lambda t: t in lang, tokens ))
    return tokens

# loads word2vec model scales the vectors and returns a dic
@memoize
def load_wv(scale=True):
    blocks2Vec = Word2Vec.load(WORD2VEC_MODEL)
    if (not scale):
        return blocks2Vec
    blocks = []
    vectors = []
    dic = {}
    for block in lang:
        try: # try if vector is available
            vectors.append(blocks2Vec[block])
            blocks.append(block)
        except Exception as e:
            continue
    # scale to have zero mean and unit standard deviation
    vectors = scale(vectors)
    for idx, block in enumerate(blocks):
        dic[block] = vectors[idx]
    return dic

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def load_tokens(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
