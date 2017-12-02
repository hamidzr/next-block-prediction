from functools import wraps
from gensim.models.word2vec import Word2Vec
import pickle
from .constants import *
from sklearn.preprocessing import scale

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
    with open('./data/uniqueBlocks.pickle', 'rb') as f:
        lang = pickle.load(f)
    for block, count in lang:
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
