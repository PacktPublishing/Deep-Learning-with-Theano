from __future__ import print_function
import numpy
from utils import load_params
from theano import theano
import theano.tensor as T
import models
import nltk

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="word", help='mode', choices=['char', 'word'])
parser.add_argument('--model', default="simple", help='model', choices=['simple', 'lstm', 'gru'])
parser.add_argument('--hidden', type=int, default=500, help='hidden layer dimension')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs of the saved model to load')
parser.add_argument('--predicts', type=int, default=10, help='number of sentences to predict')
args = parser.parse_args()

index_ = numpy.load("index_{}.npy".format(args.mode))
word_to_index = dict([(w,i) for i,w in enumerate(index_)])
print("Dictionary:")
for i in range(10): print(i, "->", index_[i])
embedding_size = len(index_)
print("Embedding size:", embedding_size)
print("Hidden layer dimension:", args.hidden)

print("Model:", args.model)
x = T.ivector()
model, params = getattr(models,args.model).model(x, embedding_size, args.hidden)
y_out = T.argmax(model, axis=-1)

print("Compiling...")
predict_model = theano.function(inputs=[x],outputs=y_out)

print("Loading parameters")
load_params("params_{}_{}_h{}_e{}".format(args.mode,args.model,args.hidden,args.epochs), params)

print("Predicting", args.predicts, "sentences")
for i in range(args.predicts):
    sentence = [0]
    words = raw_input("Type a few words:")
    print(words)
    indices = [word_to_index[word] for word in nltk.word_tokenize(words.lower())]
    sentence = sentence + indices
    print("Initial sentence",sentence)
    while sentence[-1] != 1:
        pred = predict_model(sentence)[-1]
        sentence.append(pred)
    print("Completed sentence:",sentence)
    print("->", " ".join([ index_[w] for w in sentence[1:-1]]))
