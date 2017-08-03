from __future__ import print_function
import sys
sys.path.append('..')

import os
import json
import gzip
import pickle
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import theano
import theano.tensor as T
from theano.gpuarray.basic_ops import as_gpuarray_variable, gpu_contiguous, GpuAllocEmpty, infer_context_name
from theano.gpuarray.dnn import GpuDnnConvDesc, GpuDnnConvGradI, dnn_conv

from lib import updates
from lib import inits
from lib.vis import grayscale_grid_vis
from lib.rng import py_rng, np_rng
from utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data
from lib.metrics import nnc_score, nnd_score

#### PARAMETERS
k = 1             # # of discrim updates for each gen update
l2 = 2.5e-5       # l2 weight decay
b1 = 0.5          # momentum term of adam
nc = 1            # # of channels in image
ny = 10           # # of classes
nbatch = 128      # # of examples in batch
npx = 28          # # of pixels width/height of images
nz = 100          # # of dim for Z
ngfc = 1024       # # of gen units for fully connected layers
ndfc = 1024       # # of discrim units for fully connected layers
ngf = 64          # # of gen filters in first conv layer
ndf = 64          # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 100       # # of iter at starting learning rate
niter_decay = 100 # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam


with gzip.open('/sharedfiles/mnist.pkl.gz', 'rb') as f:
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    except:
        train_set, valid_set, test_set = pickle.load(f)

trX = train_set[0]
trY = train_set[1]
vaX = valid_set[0]
vaY = valid_set[1]
teX = test_set[0]
teY = test_set[1]

ntrain, nval, ntest = len(trX), len(vaX), len(teX)

model_dir = 'models/cond_dcgan'
samples_dir = 'samples/cond_dcgan'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)


print("Building models")
gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)

gw  = gifn((nz+ny, ngfc), 'gw')
gw2 = gifn((ngfc+ny, ngf*2*7*7), 'gw2')
gw3 = gifn((ngf*2+ny, ngf, 5, 5), 'gw3')
gwx = gifn((ngf+ny, nc, 5, 5), 'gwx')

dw  = difn((ndf, nc+ny, 5, 5), 'dw')
dw2 = difn((ndf*2, ndf+ny, 5, 5), 'dw2')
dw3 = difn((ndf*2*7*7+ny, ndfc), 'dw3')
dwy = difn((ndfc+ny, 1), 'dwy')

gen_params = [gw, gw2, gw3, gwx]
discrim_params = [dw, dw2, dw3, dwy]


def batchnorm(X, g=None, b=None, u=None, s=None, a=1., e=1e-8):
    """
    batchnorm with support for not using scale and shift parameters
    as well as inference values (u and s) and partial batchnorm (via a)
    will detect and use convolutional or fully connected version
    """
    if X.ndim == 4:
        if u is not None and s is not None:
            b_u = u.dimshuffle('x', 0, 'x', 'x')
            b_s = s.dimshuffle('x', 0, 'x', 'x')
        else:
            b_u = T.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            b_s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        if a != 1:
            b_u = (1. - a)*0. + a*b_u
            b_s = (1. - a)*1. + a*b_s
        X = (X - b_u) / T.sqrt(b_s + e)
        if g is not None and b is not None:
            X = X*g.dimshuffle('x', 0, 'x', 'x') + b.dimshuffle('x', 0, 'x', 'x')
    elif X.ndim == 2:
        if u is None and s is None:
            u = T.mean(X, axis=0)
            s = T.mean(T.sqr(X - u), axis=0)
        if a != 1:
            u = (1. - a)*0. + a*u
            s = (1. - a)*1. + a*s
        X = (X - u) / T.sqrt(s + e)
        if g is not None and b is not None:
            X = X*g + b
    else:
        raise NotImplementedError
    return X


def conv_cond_concat(x, y):
    return T.concatenate([x, y*T.ones((x.shape[0], y.shape[1], x.shape[2], x.shape[3]))], axis=1)

def discrim(X, Y, w, w2, w3, wy):
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    X = conv_cond_concat(X, yb)
    h = T.nnet.relu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)), alpha=0.2 )
    h = conv_cond_concat(h, yb)
    h2 =  T.nnet.relu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))), alpha=0.2)
    h2 = T.flatten(h2, 2)
    h2 = T.concatenate([h2, Y], axis=1)
    h3 = T.nnet.relu(batchnorm(T.dot(h2, w3)))
    h3 = T.concatenate([h3, Y], axis=1)
    y = T.nnet.sigmoid(T.dot(h3, wy))
    return y


def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    img = gpu_contiguous(T.cast(X, 'float32'))
    kerns = gpu_contiguous(T.cast(w,'float32'))
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, conv_mode=conv_mode)(kerns.shape)
    out = GpuAllocEmpty(dtype='float32', context_name=infer_context_name(X))(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img


def gen(Z, Y, w, w2, w3, wx):
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    Z = T.concatenate([Z, Y], axis=1)
    h = T.nnet.relu(batchnorm(T.dot(Z, w)))
    h = T.concatenate([h, Y], axis=1)
    h2 = T.nnet.relu(batchnorm(T.dot(h, w2)))
    h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))
    h2 = conv_cond_concat(h2, yb)
    h3 = T.nnet.relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
    h3 = conv_cond_concat(h3, yb)
    x = T.nnet.sigmoid(deconv(h3, wx, subsample=(2, 2), border_mode=(2, 2)))
    return x



X = T.tensor4()
Z = T.matrix()
Y = T.matrix()

gX = gen(Z, Y, *gen_params)
p_real = discrim(X, Y, *discrim_params)
p_gen = discrim(gX, Y, *discrim_params)

d_cost_real = T.nnet.binary_crossentropy(p_real, T.ones(p_real.shape)).mean()
d_cost_gen = T.nnet.binary_crossentropy(p_gen, T.zeros(p_gen.shape)).mean()
g_cost_d = T.nnet.binary_crossentropy(p_gen, T.ones(p_gen.shape)).mean()

d_cost = d_cost_real + d_cost_gen
g_cost = g_cost_d

cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates

print('Compiling')
t = time()
_train_g = theano.function([X, Z, Y], cost, updates=g_updates)
_train_d = theano.function([X, Z, Y], cost, updates=d_updates)
_gen = theano.function([Z, Y], gX)
print('%.2f seconds to compile theano functions'%(time()-t))

tr_idxs = np.arange(len(trX))
trX_vis = np.asarray([[trX[i] for i in py_rng.sample(tr_idxs[trY==y], 20)] for y in range(10)]).reshape(200, -1)
trX_vis = trX_vis.reshape(-1, npx, npx)
grayscale_grid_vis(trX_vis, (10, 20), 'samples/cond_dcgan_etl_test.png')

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(200, nz)))
sample_ymb = floatX(OneHot(np.asarray([[i for _ in range(20)] for i in range(10)]).flatten(), ny))

def gen_samples(n, nbatch=128):
    samples = []
    labels = []
    n_gen = 0
    for i in range(n/nbatch):
        ymb = floatX(OneHot(np_rng.randint(0, 10, nbatch), ny))
        zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, nz)))
        xmb = _gen(zmb, ymb)
        samples.append(xmb)
        labels.append(np.argmax(ymb, axis=1))
        n_gen += len(xmb)
    n_left = n-n_gen
    ymb = floatX(OneHot(np_rng.randint(0, 10, n_left), ny))
    zmb = floatX(np_rng.uniform(-1., 1., size=(n_left, nz)))
    xmb = _gen(zmb, ymb)
    samples.append(xmb)
    labels.append(np.argmax(ymb, axis=1))
    return np.concatenate(samples, axis=0), np.concatenate(labels, axis=0)


print("Training")
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()
for epoch in range(1, niter+niter_decay+1):
    trX, trY = shuffle(trX, trY)
    for imb, ymb in tqdm(iter_data(trX, trY, size=nbatch), total=ntrain/nbatch):
        imb = imb.reshape(-1, nc, npx, npx)
        ymb = floatX(OneHot(ymb, ny))
        zmb = floatX(np_rng.uniform(-1., 1., size=(len(imb), nz)))
        if n_updates % (k+1) == 0:
            cost = _train_g(imb, zmb, ymb)
        else:
            cost = _train_d(imb, zmb, ymb)
        n_updates += 1
        n_examples += len(imb)
    if (epoch-1) % 5 == 0:
        g_cost = float(cost[0])
        d_cost = float(cost[1])
        gX, gY = gen_samples(100000)
        gX = gX.reshape(len(gX), -1)
        va_nnc_acc_1k = nnc_score(gX[:1000], gY[:1000], vaX, vaY, metric='euclidean')
        va_nnc_acc_10k = nnc_score(gX[:10000], gY[:10000], vaX, vaY, metric='euclidean')
        va_nnc_acc_100k = nnc_score(gX[:100000], gY[:100000], vaX, vaY, metric='euclidean')
        va_nnd_1k = nnd_score(gX[:1000], vaX, metric='euclidean')
        va_nnd_10k = nnd_score(gX[:10000], vaX, metric='euclidean')
        va_nnd_100k = nnd_score(gX[:100000], vaX, metric='euclidean')
        log = [n_epochs, n_updates, n_examples, time()-t, va_nnc_acc_1k, va_nnc_acc_10k, va_nnc_acc_100k, va_nnd_1k, va_nnd_10k, va_nnd_100k, g_cost, d_cost]
        print('%.0f %.2f %.2f %.2f %.4f %.4f %.4f %.4f %.4f'%(epoch, va_nnc_acc_1k, va_nnc_acc_10k, va_nnc_acc_100k, va_nnd_1k, va_nnd_10k, va_nnd_100k, g_cost, d_cost))

    samples = np.asarray(_gen(sample_zmb, sample_ymb))
    grayscale_grid_vis(samples.reshape(-1, npx, npx), (10, 20), 'samples/cond_dcgan/%d.png'%n_epochs)
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
    if n_epochs in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]:
        joblib.dump([p.get_value() for p in gen_params], 'models/cond_dcgan/%d_gen_params.jl'% n_epochs)
        joblib.dump([p.get_value() for p in discrim_params], 'models/cond_dcgan/%d_discrim_params.jl'% n_epochs)
