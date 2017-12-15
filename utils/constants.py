import multiprocessing

num_cores = multiprocessing.cpu_count()
INPUT = './data/scratch/scripts.txt'
BLOCK_LANG = 'data/scratch/tokens.pickle.lang'
RESULTS_DIR = './results'
FIGURES_DIR = './figures'
START_SYMBOL = 'START'
END_SYMBOL = 'END'
WORD2VEC_SIZE = 8
WORD2VEC_MODEL = './results/wv-{}.model'.format(WORD2VEC_SIZE)
