from utils import shared_uniform, shared_constant
import theano.tensor as T

def model(x, params, in_size, out_size, init_scale, bias_init=None):
    w = shared_uniform( (in_size, out_size), init_scale )
    params.append(w)
    y = T.dot(x, w)
    if (bias_init is not None):
      b = shared_constant((out_size,), bias_init)
      params.append(b)
      y += b
    return y
