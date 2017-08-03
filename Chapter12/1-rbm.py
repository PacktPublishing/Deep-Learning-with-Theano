# Simplified version of http://deeplearning.net/tutorial/rbm.html
from __future__ import print_function
import timeit
try:
    import PIL.Image as Image
except ImportError:
    import Image
import numpy
import theano
import theano.tensor as T
import os, gzip, pickle
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils import *

learning_rate=0.1
training_epochs=15
batch_size=20
n_chains=20
n_samples=10
n_hidden=500
n_visible=28 * 28
k=15

with gzip.open('/sharedfiles/mnist.pkl.gz', 'rb') as f:
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    except:
        train_set, valid_set, test_set = pickle.load(f)

test_set_x, test_set_y = shared_dataset(test_set)
train_set_x, train_set_y = shared_dataset(train_set)
n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

index = T.lscalar()
x = T.matrix('x')

rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

persistent_chain = shared_zeros((batch_size, n_hidden))

W = shared_glorot_uniform((n_visible, n_hidden), name='W')
hbias = shared_zeros(n_hidden, name='hbias')
vbias = shared_zeros(n_visible, name='vbias')
params = [W, hbias, vbias]

def sample_h_given_v(v0_sample):
    pre_sigmoid_h1 = T.dot(v0_sample, W) + hbias
    h1_mean = T.nnet.sigmoid(pre_sigmoid_h1)
    h1_sample = theano_rng.binomial(size=h1_mean.shape,n=1, p=h1_mean,dtype=theano.config.floatX)
    return [pre_sigmoid_h1, h1_mean, h1_sample]

def sample_v_given_h(h0_sample):
    pre_sigmoid_v1 = T.dot(h0_sample, W.T) + vbias
    v1_mean = T.nnet.sigmoid(pre_sigmoid_v1)
    v1_sample = theano_rng.binomial(size=v1_mean.shape,n=1, p=v1_mean,dtype=theano.config.floatX)
    return [pre_sigmoid_v1, v1_mean, v1_sample]

# pre_sigmoid_ph, ph_mean, ph_sample = sample_h_given_v(x)

# negative phase
def gibbs_hvh(h0_sample):
    pre_sigmoid_v1, v1_mean, v1_sample = sample_v_given_h(h0_sample)
    pre_sigmoid_h1, h1_mean, h1_sample = sample_h_given_v(v1_sample)
    return [pre_sigmoid_v1, v1_mean, v1_sample,
            pre_sigmoid_h1, h1_mean, h1_sample]

chain_start = persistent_chain
#chain_start = ph_sample
(
    [
        pre_sigmoid_nvs,
        nv_means,
        nv_samples,
        pre_sigmoid_nhs,
        nh_means,
        nh_samples
    ],
    updates
) = theano.scan(
    gibbs_hvh,
    outputs_info=[None, None, None, None, None, chain_start],
    n_steps=k,
    name="gibbs_hvh"
)

chain_end = nv_samples[-1]

def free_energy(v_sample):
    wx_b = T.dot(v_sample, W) + hbias
    vbias_term = T.dot(v_sample, vbias)
    hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
    return -hidden_term - vbias_term

cost = T.mean(free_energy(x)) - T.mean(free_energy(chain_end))

# We must not compute the gradient through the gibbs sampling
gparams = T.grad(cost, params, consider_constant=[chain_end])

for gparam, param in zip(gparams, params):
    updates[param] = param - gparam * T.cast(
        learning_rate,
        dtype=theano.config.floatX
    )

updates[persistent_chain] = nh_samples[-1]

# For monitoring, we compute a stochastic approximation to the pseudo-likelihood
# The pseudo-likelihood is computed with the probabilities of each bit in x
# Conditioned on all other bits
bit_i_idx = theano.shared(value=0, name='bit_i_idx')

# binarize the input image by rounding to nearest integer
xi = T.round(x)
fe_xi = free_energy(xi)

# flip bit x_i of matrix xi and preserve all other bits x_{\i}
# Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
# the result to xi_flip, instead of working in place on xi.
xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
fe_xi_flip = free_energy(xi_flip)

# equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
monitoring_cost = - T.mean(n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

# monitoring_cost = T.mean(
#     T.sum(
#         x * T.log(T.nnet.sigmoid(pre_sigmoid_nvs[-1])) +
#         (1 - x) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nvs[-1])),
#         axis=1
#     )
# )

# increment bit_i_idx % number as part of updates
updates[bit_i_idx] = (bit_i_idx + 1) % n_visible


if not os.path.isdir('rbm_plots'):
    os.makedirs('rbm_plots')
os.chdir('rbm_plots')

train_rbm = theano.function(
    [index],
    monitoring_cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size]
    },
    name='train_rbm'
)

plotting_time = 0.
start_time = timeit.default_timer()

# go through training epochs
for epoch in range(training_epochs):

    # go through the training set
    mean_cost = []
    for batch_index in range(n_train_batches):
        mean_cost += [train_rbm(batch_index)]

    print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))

    # Plot filters after each training epoch
    plotting_start = timeit.default_timer()
    # Construct image from the weight matrix
    image = Image.fromarray(
        tile_raster_images(
            X=W.get_value(borrow=True).T,
            img_shape=(28, 28),
            tile_shape=(10, 10),
            tile_spacing=(1, 1)
        )
    )
    image.save('filters_at_epoch_%i.png' % epoch)
    plotting_stop = timeit.default_timer()
    plotting_time += (plotting_stop - plotting_start)

end_time = timeit.default_timer()

pretraining_time = (end_time - start_time) - plotting_time

print('Training took %f minutes' % (pretraining_time / 60.))

#################################
#     Sampling from the RBM     #
#################################
# find out the number of test samples
number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

# pick random test examples, with which to initialize the persistent chain
test_idx = rng.randint(number_of_test_samples - n_chains)
persistent_vis_chain = theano.shared(
    numpy.asarray(
        test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
        dtype=theano.config.floatX
    )
)

plot_every = 1000
# define one step of Gibbs sampling (mf = mean-field) define a
# function that does `plot_every` steps before returning the
# sample for plotting

def gibbs_vhv(v0_sample):
    pre_sigmoid_h1, h1_mean, h1_sample = sample_h_given_v(v0_sample)
    pre_sigmoid_v1, v1_mean, v1_sample = sample_v_given_h(h1_sample)
    return [pre_sigmoid_h1, h1_mean, h1_sample,pre_sigmoid_v1, v1_mean, v1_sample]

(
    [
        presig_hids,
        hid_mfs,
        hid_samples,
        presig_vis,
        vis_mfs,
        vis_samples
    ],
    updates
) = theano.scan(
    gibbs_vhv,
    outputs_info=[None, None, None, None, None, persistent_vis_chain],
    n_steps=plot_every,
    name="gibbs_vhv"
)

# add to updates the shared variable that takes care of our persistent
# chain :.
updates.update({persistent_vis_chain: vis_samples[-1]})
# construct the function that implements our persistent chain.
# we generate the "mean field" activations for plotting and the actual
# samples for reinitializing the state of our persistent chain
sample_fn = theano.function(
    [],
    [
        vis_mfs[-1],
        vis_samples[-1]
    ],
    updates=updates,
    name='sample_fn'
)

# create a space to store the image for plotting ( we need to leave
# room for the tile_spacing as well)
image_data = numpy.zeros(
    (29 * n_samples + 1, 29 * n_chains - 1),
    dtype='uint8'
)
for idx in range(n_samples):
    # generate `plot_every` intermediate samples that we discard,
    # because successive samples in the chain are too correlated
    vis_mf, vis_sample = sample_fn()
    print(' ... plotting sample %d' % idx)
    image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
        X=vis_mf,
        img_shape=(28, 28),
        tile_shape=(1, n_chains),
        tile_spacing=(1, 1)
    )

# construct image
image = Image.fromarray(image_data)
image.save('samples.png')
