from __future__ import division
from __future__ import print_function

import argparse

import nltk
import itertools
from collections import Counter, OrderedDict, deque
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import six.moves.cPickle as pickle

import theano
import theano.tensor as T

from utils import *

unkown_token = '<UNK>'
pad_token = '<PAD>' # for padding the context

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', default='/sharedfiles/text8', type=str)
parser.add_argument('--emb_size', default=128, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--n_epochs', default=100, type=int)
parser.add_argument('--lr', default=1.0, type=float)

args = parser.parse_args()

def get_words(fname):

    words = []
    with open(fname) as fin:
      for line in fin:
        words += [w for w in line.strip().lower().split()]
    return words


def build_dictionary(words, max_df=5):

    word_freq = [[unkown_token, -1], [pad_token, 0]]
    word_freq.extend(nltk.FreqDist(itertools.chain(words)).most_common())
    word_freq = OrderedDict(word_freq)
    word2idx = {unkown_token: 0, pad_token: 1}
    idx2word = {0: unkown_token, 1: pad_token}
    idx = 2
    for w in word_freq:
      f = word_freq[w]
      if f >= max_df:
        word2idx[w] = idx
        idx2word[idx] = w
        idx += 1
      else:
        word2idx[w] = 0 # map the rare word into the unkwon token
        word_freq[unkown_token] += 1 # increment the number of unknown tokens

    return word2idx, idx2word, word_freq



def get_sample(data, data_size, word_idx, pad_token, c = 1):

  idx = max(0, word_idx - c)
  context = data[idx:word_idx]
  if word_idx + 1 < data_size:
    context += data[word_idx + 1 : min(data_size, word_idx + c + 1)]
  target = data[word_idx]
  context = [w for w in context if w != target]
  if len(context) > 0:
    return target, context + (2 * c - len(context)) * [pad_token]
  return None, None


def get_data_set(data, data_size, pad_token, c=1):
  contexts = []
  targets = []
  for i in xrange(data_size):
    target, context =  get_sample(data, data_size, i, pad_token, c)
    if not target is None:
      contexts.append(context)
      targets.append(target)

  return np.array(contexts, dtype='int32'), np.array(targets, dtype='int32')



def get_train_model(data, inputs, loss, params, batch_size=32):

    """
        trainer: Function to define the trainer of the model on the data set that bassed as the parameters of the function


        parameters:
            contexts: List of the contexts (the input of the trainer)
            targets: List of the targets.

        return:
            Theano function represents the train model
    """



    data_contexts = data[0]
    data_targets = data[1]

    context = inputs[0]
    target = inputs[1]


    learning_rate = T.fscalar('learning_rate') # theano input: the learning rate, the value of this input
                                               # can be constant like 0.1 or
                                               #it can be come from a function like a decay learning rate function





    index = T.lscalar('index') # the index of minibatch



    g_params = T.grad(cost=loss, wrt=params)

    updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(params, g_params)
    ]


    train_fun = theano.function(
        [index, learning_rate],
        loss,
        updates=updates,
        givens={
            context: data_contexts[index * args.batch_size: (index + 1) * args.batch_size],
            target: data_targets[index * args.batch_size: (index + 1) * args.batch_size]
        }
    )


    return train_fun




def get_validation_model(embeddings):

    valid_samples = T.ivector('valid_samples') # theano variable, the inpu of the validation model

    valid_embeddings = embeddings[valid_samples]

    similarity_fn = theano.function([valid_samples], T.dot(valid_embeddings, embeddings.T))

    return similarity_fn



# Step 1: Read the data into a list of strings.

words = get_words(args.data_file)
data_size = len(words)
print('Data size', data_size)

# Step 2: Build the dictionary and replace rare words with UNK token.

word2idx, idx2word, word_freq = build_dictionary(words)
data = [word2idx[w] for w in words] # map the each word to the index
del words # for reduce mem use


most_common_words = list(word_freq.items())[:5]
print('Most common words (+UNK)', most_common_words)
print('Sample data', data[:10], [idx2word[i] for i in data[:10]])
with open('idx2word.pkl', 'wb') as f:
    pickle.dump(idx2word, f, pickle.HIGHEST_PROTOCOL)

#model and training parameters

vocab_size = len(idx2word)
print("Vocabulary size", vocab_size)

#validation parameters

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(np.random.choice(valid_window, valid_size, replace=False), dtype='int32')

# Loading data

data_contexts, data_targets = get_data_set(data, data_size, word2idx[pad_token], c=2)
data_size = data_contexts.shape[0]


# make the dataset as shared for passing to GPU
data_contexts = theano.shared(data_contexts)
data_targets = theano.shared(data_targets)


#define the model, trainer and validator
from model import CBOW
[context, target], loss, params = CBOW(vocab_size, args.emb_size)
train_model = get_train_model([data_contexts, data_targets], [context, target], loss, params, args.batch_size)

embeddings = params[0]
norm = T.sqrt(T.sum(T.sqr(embeddings), axis=1, keepdims=True))
normalized_embeddings = embeddings / norm

similarity = get_validation_model(normalized_embeddings)

# Step 5: Begin training.

n_train_batches = data_size // args.batch_size
n_iters = args.n_epochs * n_train_batches
train_loss = np.zeros(n_iters)
average_loss = 0


for epoch in range(args.n_epochs):
    for minibatch_index in range(n_train_batches):

        iteration = minibatch_index + n_train_batches * epoch
        curr_loss = train_model(minibatch_index, args.lr)
        train_loss[iteration] = curr_loss
        average_loss += curr_loss


        if iteration % 2000 == 0:
            if iteration > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", iteration, ": ", average_loss)
            average_loss = 0


        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if iteration % 10000 == 0:
            sim = similarity(valid_examples)
            for i in xrange(valid_size):
                valid_word = idx2word[valid_examples[i]]
                top_k = 8 # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                  close_word = idx2word[nearest[k]]
                  log_str = "%s %s," % (log_str, close_word)
                print(log_str)

        if iteration % 100000 == 0:
            save_params("model_i{}".format(iteration), params)
