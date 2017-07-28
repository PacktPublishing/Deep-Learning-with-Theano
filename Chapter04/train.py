from __future__ import print_function
import numpy
from load import parse_text
from utils import save_params
from theano import theano
import theano.tensor as T
import timeit
import models

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default="word", help='mode', choices=['char', 'word'])
parser.add_argument('--model', default="simple", help='model', choices=['simple', 'lstm', 'gru'])
parser.add_argument('--hidden', type=int, default=500, help='hidden layer dimension')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--print_interval', type=int, default=100, help='print interval')
args = parser.parse_args()

print("Mode:", args.mode)
X_train, y_train, index_ =  parse_text("data/tiny-shakespear.txt", type = args.mode)

embedding_size = len(index_)
print("Embedding size:", embedding_size)
print("Hidden layer dimension:", args.hidden)

print("Model:", args.model)
x = T.ivector()
model, params = getattr(models,args.model).model(x, embedding_size, args.hidden)

y_out = T.argmax(model, axis=-1)

y = T.ivector()
cost = -T.mean(T.log(model)[T.arange(y.shape[0]), y])
g_params = T.grad(cost=cost, wrt=params)

lr = T.scalar('learning_rate')
updates = [
        (param, param - lr * gparam)
        for param, gparam in zip(params, g_params)
    ]


print("Compiling...")
train_model = theano.function(inputs=[x,y,lr],outputs=cost,updates=updates)

learning_rate = 0.01
n_train = len(y_train)
n_iters = args.epochs * n_train
print("Training:", args.epochs, "epochs of", n_train, "iterations")
train_loss = numpy.zeros(n_iters)

start_time = timeit.default_timer()
for epoch in range(args.epochs):
    for i in range(n_train):
        iteration = i + n_train * epoch
        train_loss[iteration] = train_model( numpy.asarray(X_train[i],dtype='int32'), numpy.asarray(y_train[i],dtype='int32') , learning_rate)
        if (len(train_loss) > 1 and train_loss[-1] > train_loss[-2]):
            learning_rate = learning_rate * 0.5
            print("Setting learning rate to {}".format(learning_rate))
        if iteration % args.print_interval == 0 :
            print('epoch {}, minibatch {}/{}, train loss {}'.format(
                epoch,
                i,
                n_train,
                train_loss[iteration]
            ))

numpy.save("train_loss_{}_{}_h{}_e{}".format(args.mode,args.model,args.hidden,args.epochs) , train_loss)
numpy.save("index_{}".format(args.mode), index_)
print("Saved index to index_{}.npy. Saved train loss to train_loss_{}_{}_h{}_e{}.npy".format(args.mode,args.mode,args.model,args.hidden,args.epochs))
save_params("params_{}_{}_h{}_e{}".format(args.mode,args.model,args.hidden,args.epochs), params)
