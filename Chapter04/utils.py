from __future__ import print_function
import numpy
from theano import theano
import theano.tensor as T

def shared_zeros(shape, dtype=theano.config.floatX, name='', n=None):
    shape = shape if n is None else (n,) + shape
    return theano.shared(numpy.zeros(shape, dtype=dtype), name=name)

def shared_glorot_uniform(shape, dtype=theano.config.floatX, name='', n=None):
    if isinstance(shape, int):
        high = numpy.sqrt(6. / shape)
    else:
        high = numpy.sqrt(6. / (numpy.sum(shape[:2]) * numpy.prod(shape[2:])))
    shape = shape if n is None else (n,) + shape
    return theano.shared(numpy.asarray(
        numpy.random.uniform(
            low=-high,
            high=high,
            size=shape),
        dtype=dtype), name=name)

def save_params(outfile, params):
    l = []
    for param in params:
        l = l + [ param.get_value() ]
    numpy.savez(outfile, *l)
    print("Saved model parameters to {}.npz".format(outfile))

def load_params(path, params):
    npzfile = numpy.load(path+".npz")
    for i, param in enumerate(params):
        param.set_value( npzfile["arr_" +str(i)] )
    print("Loaded model parameters from {}.npz".format(path))
