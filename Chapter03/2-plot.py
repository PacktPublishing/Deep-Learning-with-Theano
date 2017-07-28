from utils import *
from model import CBOW
import math
import numpy as np
import six.moves.cPickle as pickle


with open('idx2word.pkl', 'rb') as f:
    idx2word = pickle.load(f)
vocab_size = len(idx2word)
emb_size = 128
[context, target], loss, params = CBOW(vocab_size, emb_size)
load_params("model-1-epoch", params)
embeddings = params[0].get_value()
norm = math.sqrt(np.sum(np.square(embeddings), axis=1, keepdims=True)[0])
normalized_embeddings = embeddings / norm

# Step 6: Visualize the embeddings.

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(normalized_embeddings[:plot_only,:])
  labels = [idx2word[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
