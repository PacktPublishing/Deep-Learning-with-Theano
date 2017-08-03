import linear
import theano
import theano.tensor as T
from utils import get_dropout_noise, shared_zeros
from theano.ifelse import ifelse

def model(inputs, _is_training, params, batch_size, hidden_size, drop_i, drop_s, init_scale, init_H_bias, _theano_rng):
    noise_i_for_H = get_dropout_noise((batch_size, hidden_size), drop_i, _theano_rng)
    i_for_H = ifelse(_is_training, inputs * noise_i_for_H, inputs)
    i_for_H = linear.model(i_for_H, params, hidden_size, hidden_size, init_scale, bias_init=init_H_bias)

    # Dropout noise for recurrent hidden state.
    noise_s = get_dropout_noise((batch_size, hidden_size), drop_s, _theano_rng)

    def step(i_for_H_t, y_tm1, noise_s):
        s_lm1_for_H = ifelse(_is_training, y_tm1 * noise_s, y_tm1)
        return T.tanh(i_for_H_t + linear.model(s_lm1_for_H, params, hidden_size, hidden_size, init_scale))

    y_0 = shared_zeros((batch_size, hidden_size), name='h0')
    y, _ = theano.scan(step, sequences=i_for_H, outputs_info=[y_0], non_sequences = [noise_s])

    y_last = y[-1]
    sticky_state_updates = [(y_0, y_last)]

    return y, y_0, sticky_state_updates
