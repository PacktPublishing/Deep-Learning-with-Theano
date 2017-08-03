from __future__ import absolute_import, division, print_function

import time
import cPickle

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from data.reader import data_iterator, ptb_raw_data
import models.stacked as stacked
from utils import cast_floatX, get_noise_x, load_params, save_params

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="data", help='path')
parser.add_argument('--dataset', default="ptb", help='dataset', choices=['ptb', 'enwik8'])
parser.add_argument('--model', default="rhn", help='model to stack', choices=['rhn', 'rnn','lstm'])
parser.add_argument('--hidden_size', type=int, default=830, help='Hidden layer dimension')
parser.add_argument('--epochs', type=int, default=300, help='total number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--depth', type=int, default=10, help='the recurrence transition depth for rhn')
parser.add_argument('--learning_rate', type=float, default=0.2, help='learning rate')
parser.add_argument('--load_model', default="", help='load model')
parser.add_argument('--tied_embeddings', type=bool, default=True, help="use same embedding matrix for input and output word embeddings")
parser.add_argument('--tied_noise', type=bool, default=True, help="use same dropout masks for the T and H non-linearites")
config = parser.parse_args()

config.seed = 1234
config.init_scale = 0.04
config.init_T_bias = -2.0
config.lr_decay = 1.02
config.weight_decay = 1e-7
config.max_grad_norm = 10
config.num_steps = 35
config.max_epoch = 20               # number of epochs after which learning decay starts
config.drop_x = 0.25                # variational dropout rate over input word embeddings
config.drop_i = 0.75                # variational dropout rate over inputs of RHN layers(s), applied seperately in each RHN layer
config.drop_s = 0.25                # variational dropout rate over recurrent state
config.drop_o = 0.75                # variational dropout rate over outputs of RHN layer(s), applied before classification layer
config.vocab_size = 10000

print("Data loading")
train_data, valid_data, test_data, _ = ptb_raw_data(config.data_path)

print('Compiling model')
_is_training = T.iscalar('is_training')
_lr = theano.shared(cast_floatX(config.learning_rate), 'lr')
_input_data = T.imatrix('input_data')     # (batch_size, num_steps)
_noise_x = T.matrix('noise_x')            # (batch_size, num_steps)

# model
_theano_rng = RandomStreams(config.seed // 2 + 321)      # generates random numbers directly on GPU
flat_probs, params, rhn_updates, hidden_states = stacked.model(_input_data,
                                                                _noise_x,
                                                                _lr,
                                                                _is_training,
                                                                config,
                                                                _theano_rng)

# loss
_targets = T.imatrix('targets')           # (batch_size, num_steps)
flat_targets = _targets.T.flatten()
xentropies = T.nnet.categorical_crossentropy(flat_probs, flat_targets)  # (batch_size * num_steps,)
pred_loss = xentropies.sum() / config.batch_size
l2_loss = 0.5 * T.sum(T.stack([T.sum(p**2) for p in params])) # regularization
loss = pred_loss + config.weight_decay * l2_loss

# compute gradients
grads = theano.grad(loss, params)
global_grad_norm = T.sqrt(T.sum(T.stack([T.sum(g**2) for g in grads]))) # gradient clipping
clip_factor = theano.ifelse.ifelse(global_grad_norm < config.max_grad_norm,
    cast_floatX(1),
    T.cast(config.max_grad_norm / global_grad_norm, theano.config.floatX))

param_updates = [(p, p - _lr * clip_factor * g) for p, g in zip(params, grads)]
num_params = np.sum([param.get_value().size for param in params])

train = theano.function(
    [_input_data, _targets, _noise_x],
    loss,
    givens = {_is_training: np.int32(1)},
    updates = rhn_updates + param_updates)

evaluate = theano.function(
    [_input_data, _targets],
    loss,
    # Note that noise_x is unused in computation graph of this function since _is_training is false.
    givens = {_is_training: np.int32(0), _noise_x: T.zeros((config.batch_size, config.num_steps))},
    updates = rhn_updates)

print('Done. Number of parameters: %d' % num_params)

if config.load_model:
  print('Loading model...')
  load_params(config.load_model, params)


def run_epoch(data, config, is_train, verbose=False):
    """Run the model on the given data."""
    epoch_size = ((len(data) // config.batch_size) - 1) // config.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    for hidden_state in hidden_states:
      hidden_state.set_value(np.zeros_like(hidden_state.get_value()))
    for step, (x, y) in enumerate(data_iterator(data, config.batch_size, config.num_steps)):
      if is_train:
        noise_x = get_noise_x(x, config.drop_x)
        cost = train(x, y, noise_x)
      else:
        cost = evaluate(x, y)
      costs += cost
      iters += config.num_steps
      if verbose and step % (epoch_size // 10) == 10:
        print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / epoch_size, np.exp(costs / iters),
                                                         iters * config.batch_size / (time.time() - start_time)))
    return np.exp(costs / iters)


trains, vals, tests, best_val, save_path = [np.inf], [np.inf], [np.inf], np.inf, None

for i in range(config.epochs):
    lr_decay = config.lr_decay ** max(i - config.max_epoch + 1, 0.0)
    _lr.set_value(cast_floatX(config.learning_rate / lr_decay))

    print("Epoch: %d Learning rate: %.3f" % (i + 1, _lr.get_value()))

    train_perplexity = run_epoch(train_data, config, is_train=True, verbose=True)
    print("Epoch: %d Train Perplexity: %.3f, Bits: %.3f" % (i + 1, train_perplexity, np.log2(train_perplexity)))

    valid_perplexity = run_epoch(valid_data, config, is_train=False)
    print("Epoch: %d Valid Perplexity (batched): %.3f, Bits: %.3f" % (i + 1, valid_perplexity, np.log2(valid_perplexity)))

    test_perplexity = run_epoch(test_data, config, is_train=False)
    print("Epoch: %d Test Perplexity (batched): %.3f, Bits: %.3f" % (i + 1, test_perplexity, np.log2(test_perplexity)))

    trains.append(train_perplexity)
    vals.append(valid_perplexity)
    tests.append(test_perplexity)

    if valid_perplexity < best_val:
      best_val = valid_perplexity
      print("Best Batched Valid Perplexity improved to %.03f" % best_val)
      save_params('./theano_rhn_' + config.dataset + '_' + str(config.seed) + '_best_model.pkl', params)

    print("Training is over.")
    best_val_epoch = np.argmin(vals)
    print("Best Batched Validation Perplexity %.03f (Bits: %.3f) was at Epoch %d" %
        (vals[best_val_epoch], np.log2(vals[best_val_epoch]), best_val_epoch))
    print("Training Perplexity at this Epoch was %.03f, Bits: %.3f" %
        (trains[best_val_epoch], np.log2(trains[best_val_epoch])))
    print("Batched Test Perplexity at this Epoch was %.03f, Bits: %.3f" %
        (tests[best_val_epoch], np.log2(tests[best_val_epoch])))
