import multiprocessing

num_cores = multiprocessing.cpu_count()
INPUT = './data/cleanSample.txt'
RESULTS_DIR = './results'
START_SYMBOL = 'START'
END_SYMBOL = 'END'
WORD2VEC_MODEL = './data/block2vec10.model'
WORD2VEC_SIZE = 10
