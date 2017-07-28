import numpy
from utils import shared_zeros,shared_glorot_uniform
from theano import theano
import theano.tensor as T

def model(x, embedding_size, n_hidden):

    # hidden and input weights
    U = shared_glorot_uniform(( embedding_size,n_hidden), name="U")
    W = shared_glorot_uniform((n_hidden, n_hidden), name="W")
    bh = shared_zeros((n_hidden,), name="bh")

    # output weights
    V = shared_glorot_uniform(( n_hidden, embedding_size), name="V")
    by = shared_zeros((embedding_size,), name="by")

    params = [U,V,W,by,bh]

    def step(x_t, h_tm1):
        h_t = T.tanh(U[x_t] + T.dot( h_tm1, W) + bh)
        y_t = T.dot(h_t, V) + by
        return h_t, y_t

    h0 = shared_zeros((n_hidden,), name='h0')
    [h, y_pred], _ = theano.scan(step, sequences=x, outputs_info=[h0, None], truncate_gradient=10)

    model = T.nnet.softmax(y_pred)
    return model, params
