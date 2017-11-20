#!/usr/bin/env python3.6
from tqdm import tqdm #progress bar
import pandas as pd
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
from bokeh.plotting import figure, show, output_file

parser = argparse.ArgumentParser()
parser.add_argument('-t', "--tokenize", help="retokenize ?",
                    action="store_true")
parser.add_argument('-m', "--mode", required=True, help="working mode, train or evaluate")
parser.add_argument("--tokens_path", required=True, help="tokens pickle file path")
parser.add_argument("--model_path", required=True, help="model file path")
parser.add_argument("--plot_path", required=True, help="plot path / name")
parser.add_argument("--block", help="find similar blocks")
args = parser.parse_args()

print(args)

# reads prjoects in
def load_data(path):
    # TODO separate by sprites ? 
    df = pd.read_csv(path, header=None);
    df.set_axis(['text'], axis='columns', inplace=True)
    print(list(df.columns)) # columns
    print(df.head()) # columns
    print(df.describe())
    return df

# cleansout and tokenizes one sentence or project 
def tokenize(comment):
    try:
        # comment = unicode(comment.decode('utf-8').lower())
        comment = unicode(comment.decode('utf-8'))
        tokens = word_tokenize(comment)
    except Exception as e:
        print(e)
        tokens = 'BADPROJECT'
    return tokens

# processes all the sentences and removed invalid ones
def process_sentences(dataFrame):
    dataFrame['tokens'] = dataFrame.text.progress_map(tokenize)
    dataFrame['token-count'] = dataFrame.tokens.progress_map(lambda tokens: len(tokens))
    dataFrame = dataFrame[dataFrame.tokens != 'BADPROJECT']
    # dataFrame.drop('index', inplace=True, axis='columns')
    return dataFrame


if (args.mode == 'train' and args.tokenize):
    df = load_data('~/datasets/blocks/project-sentences.txt')
    print 'tokenizing..'
    df = process_sentences(df)
    df.to_pickle(args.tokens_path)
    print 'finished tokenizing'
else:
    df = pd.read_pickle(args.tokens_path)

x_train = df['tokens'].tolist()

# split data
# x_train, x_test, y_train, y_test = train_test_split(np.array(data),
#                                                     np.full(len(data), 0), test_size=0.01)

if (args.mode == 'train'):
    print('training..')
    # set vector size and min block count
    blocks2Vec = Word2Vec(size=10, min_count=10)
    blocks2Vec.build_vocab([x for x in tqdm(x_train)])
    blocks2Vec.train([x for x in tqdm(x_train)], total_examples=len(x_train), epochs=blocks2Vec.iter)
    blocks2Vec.save(args.model_path)
else:
    print('loading model..')
    blocks2Vec = Word2Vec.load(args.model_path)

if (args.block):
    block = args.block
    print blocks2Vec[block]
    print blocks2Vec.most_similar(block)


# defining the chart
output_file(args.plot_path)
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="block vectors",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

print('number of words #', len(blocks2Vec.wv.vocab.keys()))
# getting a list of word vectors. limit to 10000. each is of 200 dimensions
block_vectors = [blocks2Vec[w] for w in blocks2Vec.wv.vocab.keys()]

# dimensionality reduction. converting the vectors to 2d vectors
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, n_iter=1000)
print('finding the best 2d representation..')
tsne_w2v = tsne_model.fit_transform(block_vectors)
# putting everything in a dataframe
tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])

tsne_df['blocks'] = blocks2Vec.wv.vocab.keys()

# plotting. the corresponding word appears when you hover on the data point.
plot_tfidf.scatter(x='x', y='y', source=tsne_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"block": "@blocks"}
show(plot_tfidf)
