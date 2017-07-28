import numpy
from utils import shared_zeros,shared_glorot_uniform
from theano import theano
import theano.tensor as T

def model(x, embedding_size, n_hidden):

    # input gate
    W_xi = shared_glorot_uniform(( embedding_size,n_hidden))
    W_hi = shared_glorot_uniform(( n_hidden,n_hidden))
    W_ci = shared_glorot_uniform(( n_hidden,n_hidden))
    b_i = shared_zeros((n_hidden,))

    # forget gate
    W_xf = shared_glorot_uniform(( embedding_size, n_hidden))
    W_hf = shared_glorot_uniform(( n_hidden,n_hidden))
    W_cf = shared_glorot_uniform(( n_hidden,n_hidden))
    b_f = shared_zeros((n_hidden,))

    # output gate
    W_xo = shared_glorot_uniform(( embedding_size, n_hidden))
    W_ho = shared_glorot_uniform(( n_hidden,n_hidden))
    W_co = shared_glorot_uniform(( n_hidden,n_hidden))
    b_o = shared_zeros((n_hidden,))

    # cell weights
    W_xc = shared_glorot_uniform(( embedding_size, n_hidden))
    W_hc = shared_glorot_uniform(( n_hidden,n_hidden))
    b_c = shared_zeros((n_hidden,))

    # output weights
    W_y = shared_glorot_uniform(( n_hidden, embedding_size), name="V")
    b_y = shared_zeros((embedding_size,), name="by")

    params = [W_xi,W_hi,W_ci,b_i,W_xf,W_hf,W_cf,b_f,W_xo,W_ho,W_co,b_o,W_xc,W_hc,b_c,W_y,b_y]

    def step(x_t, h_tm1, c_tm1):
        i_t = T.nnet.sigmoid(W_xi[x_t] + T.dot(W_hi, h_tm1) + T.dot(W_ci, c_tm1) + b_i)
        f_t = T.nnet.sigmoid(W_xf[x_t] + T.dot(W_hf, h_tm1) + T.dot(W_cf, c_tm1) + b_f)
        c_t = f_t * c_tm1 + i_t * T.tanh(W_xc[x_t] + T.dot(W_hc, h_tm1) + b_c)
        o_t = T.nnet.sigmoid(W_xo[x_t] + T.dot(W_ho, h_tm1) + T.dot(W_co, c_t) + b_o)
        h_t = o_t * T.tanh(c_t)
        y_t = T.dot(h_t, W_y) + b_y
        return h_t, c_t, y_t

    h0 = shared_zeros((n_hidden,), name='h0')
    c0 = shared_zeros((n_hidden,), name='c0')
    [h, c, y_pred], _ = theano.scan(step, sequences=x, outputs_info=[h0, c0, None], truncate_gradient=10)

    model = T.nnet.softmax(y_pred)
    return model, params
