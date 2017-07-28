from __future__ import print_function
import numpy

def save_params(outfile, params):
    l = []
    for param in params:
        l = l + [ param.get_value() ]
    numpy.savez(outfile+".npz", *l)
    print("Saved model parameters to {}.npz".format(outfile))

def load_params(path, params):
    npzfile = numpy.load(path+".npz")
    for i, param in enumerate(params):
        param.set_value( npzfile["arr_" +str(i)] )
    print("Loaded model parameters from {}.npz".format(path))
