#!/usr/bin/env python3.6
from tqdm import tqdm #progress bar
from utils import constants, helpers
import pandas as pd
import time
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split # helps split the dataset
tqdm.pandas(desc='progress-bar')
import argparse
# import visualization help
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from matplotlib import pyplot as plt
from bokeh.plotting import figure, show, output_file

parser = argparse.ArgumentParser()
parser.add_argument('-m', "--mode", required=True, help="working mode, train or evaluate")
# parser.add_argument("--tokens_path", required=True, help="tokens pickle file path")
parser.add_argument("--vec_size", type=int, required=True, help="dense vector size")
parser.add_argument("--model_path", required=False, help="model path to load and evaluate")
# parser.add_argument("--plot_path", required=True, help="plot path / name")
parser.add_argument("--block", help="find similar blocks")
args = parser.parse_args()

print(args)

# reads prjoects in
def load_data(path):
    # TODO separate by sprites ? 
    print('loading..')
    df = pd.read_csv(path, header=None);
    df.set_axis(['text'], axis='columns', inplace=True)
    print(list(df.columns)) # columns
    print(df.head()) # columns
    print(df.describe())
    return df

# cleansout and tokenizes one sentence or project 
def tokenize(script):
    try:
        # script = unicode(script.decode('utf-8'))
        tokens = helpers.script_tokenizer(script)
    except Exception as e:
        print(e)
        tokens = 'BADPROJECT'
    return tokens

# processes all the scripts and removes invalid ones
def process_sentences(dataFrame):
    dataFrame['tokens'] = dataFrame.text.progress_map(tokenize)
    dataFrame['token-count'] = dataFrame.tokens.progress_map(lambda tokens: len(tokens))
    dataFrame = dataFrame[dataFrame.tokens != 'BADPROJECT']
    # dataFrame.drop('index', inplace=True, axis='columns')
    return dataFrame


# fits a tSNE representation
def visualize_embeddings(wvModel):
    # getting a list of word vectors
    block_vectors = [wvModel[w] for w in wvModel.wv.vocab.keys()]

    # dimensionality reduction. converting the vectors to 2d vectors
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, n_iter=1000)
    print('fitting the best 2d representation..')
    tsne_w2v = tsne_model.fit_transform(block_vectors)
    # putting everything in a dataframe
    tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])

    tsne_df['blocks'] = wvModel.wv.vocab.keys()
    return tsne_df

def plot_bokeh(df):
    # defining the chart
    output_file(f'{constants.FIGURES_DIR}/wv-{args.vec_size}-{int(time.time())}.html')
    plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="block vectors",
        tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
        x_axis_type=None, y_axis_type=None, min_border=1)

    # plotting. corresponding word appears when you hover on the data point.
    plot_tfidf.scatter(x='x', y='y', source=df)
    hover = plot_tfidf.select(dict(type=HoverTool))
    hover.tooltips={"block": "@blocks"}
    show(plot_tfidf)

def plot_image(df):
    filename = f'{constants.FIGURES_DIR}/wv-{args.vec_size}-{int(time.time())}.png'
    assert len(df.x) != len(df.blocks), "mismatching labels and coordinates"
    xs = df.x.tolist()
    ys = df.y.tolist()
    labels = df['blocks'].tolist()
    plt.figure(figsize=(18, 18))    # in inches
    for i, label in enumerate(labels):
        x = xs[i]
        y = ys[i]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)


def interactive_test(model):
    while True:
        block = input('pass in a block: ')
        try:
            print('corresponding vec', model[block])
            print(model.most_similar(block))
            # seq = input('pass in a script: ').split(' ')
            # print('out of place block: ', model.doesnt_match(seq))
        except Exception as e:
            print('invalid input')

def similarities(wvModel):
    blocks = wvModel.wv.vocab.keys()
    stats = []
    for b in blocks:
        pairs = wvModel.most_similar(b)
        clusterScore = 0
        for w, p in pairs[:3]:
            clusterScore += p
        stats.append((b, clusterScore))
    stats.sort(key=lambda x: x[1])
    print(stats)

# if (args.mode == 'train'):
#     df = load_data(constants.INPUT)
#     print('tokenizing..')
#     df = process_sentences(df)
#     # df.to_pickle(args.tokens_path)
#     print('finished tokenizing')
# else:
#     print('Loading tokens..')
#     df = pd.read_pickle(args.tokens_path)

# x_train = df['tokens'].tolist()

# x_train = helpers.load_tokens(args.tokens_path)
with open(constants.INPUT, 'r') as f:
    inputLines = f.read().split('\n')
x_train = [helpers.script_tokenizer(script) for script in inputLines]


if (args.mode == 'train'):
    # set vector size and min block count
    # using CBOW
    blocks2Vec = Word2Vec(size=args.vec_size, window=2, min_count=10, sg=0, iter=10)
    print('building vocab.. ')
    blocks2Vec.build_vocab([x for x in tqdm(x_train)])
    print('training..')
    blocks2Vec.train([x for x in tqdm(x_train)], total_examples=len(x_train), epochs=blocks2Vec.iter)
    print('saving model..')
    blocks2Vec.save(f'{constants.RESULTS_DIR}/wv-{args.vec_size}-{int(time.time())}.model')
else:
    print('loading model..')
    blocks2Vec = Word2Vec.load(args.model_path)

if (args.mode == 'eval'):
    interactive_test(blocks2Vec)
else:
    df = visualize_embeddings(blocks2Vec)
    plot_bokeh(df)
    plot_image(df)
