from __future__ import absolute_import, division, print_function

import numpy as np

import theano
import theano.tensor as T
from theano.ifelse import ifelse

from models import rhn
from models import rnn
from models import lstm

from utils import shared_uniform, get_dropout_noise,  shared_zeros, cast_floatX

floatX = theano.config.floatX

def model(_input_data, _noise_x, _lr, _is_training, config, _theano_rng):
    embedding = shared_uniform(( config.vocab_size,config.hidden_size), config.init_scale)
    params = [embedding]

    inputs = embedding[_input_data.T]          # (num_steps, batch_size, hidden_size)
    inputs = ifelse(_is_training, inputs * T.shape_padright(_noise_x.T), inputs)

    rhn_updates = []
    sticky_hidden_states = []                               # shared variables which are reset before each epoch
    for _ in range(config.num_layers):
        # y shape: (num_steps, batch_size, hidden_size)
        if config.model == "rhn":
            print("  with RHN cell")
            y, y_0, sticky_state_updates = rhn.model(
                inputs, _is_training, params,
                config.depth, config.batch_size, config.hidden_size,
                config.drop_i, config.drop_s,
                config.init_scale, config.init_T_bias,config.init_scale,
                config.tied_noise,
                _theano_rng)
        elif config.model == "lstm":
            print("  with LSTM cell")
            y, y_0, sticky_state_updates = lstm.model(
                inputs, _is_training, params,
                config.batch_size, config.hidden_size,
                config.drop_i, config.drop_s,
                config.init_scale, config.init_scale,
                config.tied_noise,
                _theano_rng)
        else:
            print("  with RNN cell")
            y, y_0, sticky_state_updates = rnn.model(
                inputs, _is_training, params,
                config.batch_size, config.hidden_size,
                config.drop_i, config.drop_s,
                config.init_scale, config.init_scale,
                _theano_rng)
        rhn_updates += sticky_state_updates
        inputs = y
        # The recurrent hidden state of the RHN is sticky (the last hidden state of one batch is carried over to the next batch,
        # to be used as an initial hidden state).  These states are kept in shared variables and are reset before every epoch.
        sticky_hidden_states.append(y_0)

    noise_o = get_dropout_noise((config.batch_size, config.hidden_size), config.drop_o, _theano_rng)
    outputs = ifelse(_is_training, y * T.shape_padleft(noise_o), y)               # (num_steps, batch_size, hidden_size)

    # logits
    if config.tied_embeddings:
        softmax_w = embedding.T
    else:
        softmax_w = shared_uniform((config.hidden_size, config.vocab_size), config.init_scale)
        params = params + [softmax_w]

    softmax_b = shared_uniform((config.vocab_size,), config.init_scale)
    params = params + [softmax_b]

    logits = T.dot(outputs, softmax_w) + softmax_b                          # (num_steps, batch_size, vocab_size)

    # probabilities and prediction loss
    flat_logits = logits.reshape((config.batch_size * config.num_steps, config.vocab_size))
    flat_probs = T.nnet.softmax(flat_logits)

    return flat_probs, params, rhn_updates, sticky_hidden_states
