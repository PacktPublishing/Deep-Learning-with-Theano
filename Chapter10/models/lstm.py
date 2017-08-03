import linear
import theano
import theano.tensor as T
from utils import get_dropout_noise, shared_zeros
from theano.ifelse import ifelse

def model(inputs, _is_training, params, batch_size, hidden_size, drop_i, drop_s, init_scale, init_H_bias, tied_noise, _theano_rng):
    noise_i_for_i = get_dropout_noise((batch_size, hidden_size), drop_i, _theano_rng)
    noise_i_for_f = get_dropout_noise((batch_size, hidden_size), drop_i, _theano_rng) if not tied_noise else noise_i_for_i
    noise_i_for_c = get_dropout_noise((batch_size, hidden_size), drop_i, _theano_rng) if not tied_noise else noise_i_for_i
    noise_i_for_o = get_dropout_noise((batch_size, hidden_size), drop_i, _theano_rng) if not tied_noise else noise_i_for_i

    i_for_i = ifelse(_is_training, inputs* noise_i_for_i, inputs)
    i_for_f = ifelse(_is_training, inputs* noise_i_for_f, inputs)
    i_for_c = ifelse(_is_training, inputs* noise_i_for_c, inputs)
    i_for_o = ifelse(_is_training, inputs* noise_i_for_o, inputs)

    i_for_i = linear.model(i_for_i, params, hidden_size, hidden_size, init_scale, bias_init=init_H_bias)
    i_for_f = linear.model(i_for_f, params, hidden_size, hidden_size, init_scale, bias_init=init_H_bias)
    i_for_c = linear.model(i_for_c, params, hidden_size, hidden_size, init_scale, bias_init=init_H_bias)
    i_for_o = linear.model(i_for_o, params, hidden_size, hidden_size, init_scale, bias_init=init_H_bias)

    # Dropout noise for recurrent hidden state.
    noise_s = get_dropout_noise((batch_size, hidden_size), drop_s, _theano_rng)
    if not tied_noise:
      noise_s = T.stack(noise_s, get_dropout_noise((batch_size, hidden_size), drop_s, _theano_rng),
        get_dropout_noise((batch_size, hidden_size), drop_s, _theano_rng),
        get_dropout_noise((batch_size, hidden_size), drop_s, _theano_rng))


    def step(i_for_i_t,i_for_f_t,i_for_c_t,i_for_o_t, y_tm1, c_tm1, noise_s):
        noise_s_for_i = noise_s if tied_noise else noise_s[0]
        noise_s_for_f = noise_s if tied_noise else noise_s[1]
        noise_s_for_c = noise_s if tied_noise else noise_s[2]
        noise_s_for_o = noise_s if tied_noise else noise_s[3]

        s_lm1_for_i = ifelse(_is_training, y_tm1 * noise_s_for_i, y_tm1)
        s_lm1_for_f = ifelse(_is_training, y_tm1 * noise_s_for_f, y_tm1)
        s_lm1_for_c = ifelse(_is_training, y_tm1 * noise_s_for_c, y_tm1)
        s_lm1_for_o = ifelse(_is_training, y_tm1 * noise_s_for_o, y_tm1)

        i_t = T.nnet.sigmoid(i_for_i_t + linear.model(s_lm1_for_i, params, hidden_size, hidden_size, init_scale))
        f_t = T.nnet.sigmoid(i_for_o_t + linear.model(s_lm1_for_f, params, hidden_size, hidden_size, init_scale))
        c_t = f_t * c_tm1 + i_t * T.tanh(i_for_c_t + linear.model(s_lm1_for_c, params, hidden_size, hidden_size, init_scale))
        o_t = T.nnet.sigmoid(i_for_o_t + linear.model(s_lm1_for_o, params, hidden_size, hidden_size, init_scale))
        return o_t * T.tanh(c_t), c_t

    y_0 = shared_zeros((batch_size,hidden_size), name='h0')
    c_0 = shared_zeros((batch_size,hidden_size), name='c0')
    [y, c], _ = theano.scan(step,
        sequences=[i_for_i,i_for_f,i_for_c,i_for_o],
        outputs_info=[y_0,c_0],
        non_sequences = [noise_s])

    y_last = y[-1]
    sticky_state_updates = [(y_0, y_last)]

    return y, y_0, sticky_state_updates
