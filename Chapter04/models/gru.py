import numpy
from utils import shared_zeros,shared_glorot_uniform
from theano import theano
import theano.tensor as T

def model(x, embedding_size, n_hidden):

    # Update gate weights
    W_xz = shared_glorot_uniform(( embedding_size,n_hidden))
    W_hz = shared_glorot_uniform(( n_hidden,n_hidden))
    b_z = shared_zeros((n_hidden,))

    # Reset gate weights
    W_xr = shared_glorot_uniform(( embedding_size,n_hidden))
    W_hr = shared_glorot_uniform(( n_hidden,n_hidden))
    b_r = shared_zeros((n_hidden,))

    # Hidden layer
    W_xh = shared_glorot_uniform(( embedding_size,n_hidden))
    W_hh = shared_glorot_uniform(( n_hidden,n_hidden))
    b_h = shared_zeros((n_hidden,))

    # Output weights
    W_y = shared_glorot_uniform(( n_hidden, embedding_size), name="V")
    b_y = shared_zeros((embedding_size,), name="by")

    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_y, b_y]

    def step(x_t, h_tm1):
        z_t = T.nnet.sigmoid(W_xz[x_t] + T.dot(W_hz, h_tm1) + b_z)
        r_t = T.nnet.sigmoid(W_xr[x_t] + T.dot(W_hr, h_tm1) + b_r)
        can_h_t = T.tanh(W_xh[x_t] + r_t * T.dot(W_hh, h_tm1) + b_h)
        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t
        y_t = T.dot(h_t, W_y) + b_y
        return h_t, y_t

    h0 = shared_zeros((n_hidden,), name='h0')
    [h, y_pred], _ = theano.scan(step, sequences=x, outputs_info=[h0, None], truncate_gradient=10)

    model = T.nnet.softmax(y_pred)
    return model, params
