from linear import model as linear
import theano
import theano.tensor as T
from utils import get_dropout_noise, shared_zeros
from theano.ifelse import ifelse

def model(inputs, _is_training, params, depth, batch_size, hidden_size, drop_i, drop_s, init_scale, init_T_bias, init_H_bias, tied_noise, _theano_rng):
  noise_i_for_H = get_dropout_noise((batch_size, hidden_size), drop_i, _theano_rng)
  noise_i_for_T = get_dropout_noise((batch_size, hidden_size), drop_i, _theano_rng) if not tied_noise else noise_i_for_H

  i_for_H = ifelse(_is_training, noise_i_for_H * inputs, inputs)
  i_for_T = ifelse(_is_training, noise_i_for_T * inputs, inputs)

  i_for_H = linear(i_for_H, params, in_size=hidden_size, out_size=hidden_size, init_scale=init_scale, bias_init=init_H_bias)
  i_for_T = linear(i_for_T, params, in_size=hidden_size, out_size=hidden_size, init_scale=init_scale, bias_init=init_T_bias)

  # Dropout noise for recurrent hidden state.
  noise_s = get_dropout_noise((batch_size, hidden_size), drop_s, _theano_rng)
  if not tied_noise:
    noise_s = T.stack(noise_s, get_dropout_noise((batch_size, hidden_size), drop_s, _theano_rng))

  def deep_step_fn(i_for_H_t, i_for_T_t, y_tm1, noise_s):
    tanh, sigm = T.tanh, T.nnet.sigmoid
    noise_s_for_H = noise_s if tied_noise else noise_s[0]
    noise_s_for_T = noise_s if tied_noise else noise_s[1]

    s_lm1 = y_tm1
    for l in range(depth):
      s_lm1_for_H = ifelse(_is_training, s_lm1 * noise_s_for_H, s_lm1)
      s_lm1_for_T = ifelse(_is_training, s_lm1 * noise_s_for_T, s_lm1)
      if l == 0:
        # On the first micro-timestep of each timestep we already have bias
        # terms summed into i_for_H_t and into i_for_T_t.
        H = tanh(i_for_H_t + linear(s_lm1_for_H, params, in_size=hidden_size, out_size=hidden_size, init_scale=init_scale))
        Tr = sigm(i_for_T_t + linear(s_lm1_for_T, params, in_size=hidden_size, out_size=hidden_size, init_scale=init_scale))
      else:
        H = tanh(linear(s_lm1_for_H, params, in_size=hidden_size, out_size=hidden_size, init_scale=init_scale, bias_init=init_H_bias))
        Tr = sigm(linear(s_lm1_for_T, params, in_size=hidden_size, out_size=hidden_size, init_scale=init_scale, bias_init=init_T_bias))
      s_l = (H - s_lm1) * Tr + s_lm1
      s_lm1 = s_l

    y_t = s_l
    return y_t

  y_0 = shared_zeros((batch_size, hidden_size))

  y, _ = theano.scan(deep_step_fn,
    sequences = [i_for_H, i_for_T],
    outputs_info = [y_0],
    non_sequences = [noise_s])

  y_last = y[-1]
  sticky_state_updates = [(y_0, y_last)]

  return y, y_0, sticky_state_updates
