import theano
import theano.tensor as T
import numpy as np
import math

def CBOW(vocab_size, emb_size):

    """
    CBOW: Function to define the CBOW model


    parameters:
        vocab_size: the vocabulary size
        emb_size: dimension of the embedding vector

    return:
        List of theano variables [context, target], represents the model input,
        Theano function represents the loss (i.e. the cose or the objective) function,
        List of theano (shared) variable params, represents the parameters of the model.
    """

    context = T.imatrix(name='context')
    target = T.ivector('target')

    W_in_values = np.asarray(np.random.uniform(-1.0, 1.0, (vocab_size, emb_size)),
                                dtype=theano.config.floatX)


    W_out_values = np.asarray(np.random.normal(scale=1.0 / math.sqrt(emb_size), size=(emb_size, vocab_size)),
                                dtype=theano.config.floatX)
    W_in = theano.shared(
            value=W_in_values,
            name='W_in',
            borrow=True)

    W_out = theano.shared(
            value=W_out_values,
            name='W_out',
            borrow=True)


    h = T.mean(W_in[context], axis=1) # compute the hidden (projection) layer output : input -> hidden (eq. 1)
    uj = T.dot(h, W_out) # hidden -> output (eq. 2)
    p_target_given_contex = T.nnet.softmax(uj) # softmax activation (eq. 3)
    loss = -T.mean(T.log(p_target_given_contex)[T.arange(target.shape[0]), target]) # loss function (eq. 4)

    params = [W_in, W_out]

    return [context, target], loss, params
