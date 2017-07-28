from __future__ import print_function
from __future__ import division
import numpy as np
import theano.tensor as T
import theano
import lasagne
from confusionmatrix import ConfusionMatrix
import math
import mnist_cnn

import argparse
np.random.seed(1234)
parser = argparse.ArgumentParser()
parser.add_argument("-lr", type=float, default=0.0005)
parser.add_argument("-decayinterval", type=int, default=10)
parser.add_argument("-decayfac", type=float, default=1.5)
parser.add_argument("-nodecay", type=int, default=30)
parser.add_argument("-optimizer", type=str, default='rmsprop')
parser.add_argument("-dropout", type=float, default=0.0)
parser.add_argument("-downsample", type=float, default=3.0)
args = parser.parse_args()
print('#'*80)
for name, val in sorted(vars(args).items()):
    print("#{}{}{}".format(name, " "*(35 - len(name)), val))
print('#'*80)

np.random.seed(123)
TOL = 1e-5
batch_size = 100
num_rnn_units = 256
num_classes = 10
NUM_EPOCH = 300
MONITOR = False
MAX_NORM = 5.0
LOOK_AHEAD = 50

print("Loading data")
data = np.load("mnist_sequence3_sample_8distortions_9x9.npz")
dim = int(math.sqrt(data['X_train'].shape[1]))
x_train, y_train = data['X_train'].reshape((-1, dim, dim)), data['y_train']
x_valid, y_valid = data['X_valid'].reshape((-1, dim, dim)), data['y_valid']
x_test, y_test = data['X_test'].reshape((-1, dim, dim)), data['y_test']
n_batches_train = x_train.shape[0] // batch_size
n_batches_valid = x_valid.shape[0] // batch_size
num_steps = y_train.shape[1]

print("Building model")
## define dropout as a symbolic variable
sh_drp = theano.shared(lasagne.utils.floatX(args.dropout))

l_in = lasagne.layers.InputLayer((None, dim, dim))
l_dim = lasagne.layers.DimshuffleLayer(l_in, (0, 'x', 1, 2))
l_pool0_loc = lasagne.layers.MaxPool2DLayer(l_dim, pool_size=(2, 2))
l_conv2_loc = mnist_cnn.model(l_pool0_loc, input_dim=dim, p=sh_drp, num_units=0)

class Repeat(lasagne.layers.Layer):
    def __init__(self, incoming, n, **kwargs):
        super(Repeat, self).__init__(incoming, **kwargs)
        self.n = n

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0], self.n] + list(input_shape[1:]))

    def get_output_for(self, input, **kwargs):
        tensors = [input]*self.n
        stacked = theano.tensor.stack(*tensors)
        dim = [1, 0] + range(2, input.ndim+1)
        return stacked.dimshuffle(dim)

l_repeat_loc = Repeat(l_conv2_loc, n=num_steps)
l_gru = lasagne.layers.GRULayer(l_repeat_loc, num_units=num_rnn_units,
                                unroll_scan=True)

l_shp = lasagne.layers.ReshapeLayer(l_gru, (-1, num_rnn_units))

b = np.zeros((2, 3), dtype=theano.config.floatX)
b[0, 0] = 1.0
b[1, 1] = 1.0

l_A_net = lasagne.layers.DenseLayer(
    l_shp,
    num_units=6,
    name='A_net',
    b=b.flatten(),
    W=lasagne.init.Constant(0.0),
    nonlinearity=lasagne.nonlinearities.identity)

l_conv_to_transform = lasagne.layers.ReshapeLayer(
    Repeat(l_dim, n=num_steps), [-1] + list(l_dim.output_shape[-3:]))

l_transform = lasagne.layers.TransformerLayer(
    incoming=l_conv_to_transform,
    localization_network=l_A_net,
    downsample_factor=args.downsample)

l_out = mnist_cnn.model(l_transform, input_dim=dim, p=sh_drp, num_units=400)

# Build graph
x = T.tensor3()
output_train = lasagne.layers.get_output(l_out, x, deterministic=False)
output_eval, l_A_eval = lasagne.layers.get_output([l_out, l_A_net],
                                        x, deterministic=True)

# cost
output_flat = T.reshape(output_train, (-1, num_classes))
y = T.imatrix()
cost = T.nnet.categorical_crossentropy(output_flat+TOL, y.flatten())
cost = T.mean(cost)

# params
all_params = lasagne.layers.get_all_params(l_out, trainable=True)
trainable_params = lasagne.layers.get_all_params(l_out, trainable=True)
# for p in trainable_params:
#     print(p.name)

# update
all_grads = T.grad(cost, trainable_params)
all_grads = [T.clip(g, -1.0, 1.0) for g in all_grads]
sh_lr = theano.shared(lasagne.utils.floatX(args.lr))
# adam works with lr 0.001
updates, norm = lasagne.updates.total_norm_constraint(
    all_grads, max_norm=MAX_NORM, return_norm=True)

if args.optimizer == 'rmsprop':
    updates = lasagne.updates.rmsprop(updates, trainable_params,
            learning_rate=sh_lr)
elif args.optimizer == 'adam':
    updates = lasagne.updates.adam(updates, trainable_params,
            learning_rate=sh_lr)

print("Compiling")
if MONITOR:
    f_train = theano.function([x, y], [cost, output_train, norm]
                    + all_grads + updates.values(),updates=updates)
else:
    f_train = theano.function([x, y], [cost, output_train, norm],
                    updates=updates)

f_eval = theano.function([x],
                [output_eval, l_A_eval.reshape((-1, num_steps, 6))])

print("Training")
best_valid = 0
look_count = LOOK_AHEAD
cost_train_lst = []
last_decay = 0
for epoch in range(NUM_EPOCH):
    # eval train
    shuffle = np.random.permutation(x_train.shape[0])

    if epoch < 5:
        sh_drp.set_value(lasagne.utils.floatX((epoch)*args.dropout/5.0))
    else:
        sh_drp.set_value(lasagne.utils.floatX(args.dropout))

    for i in range(n_batches_train):
        idx = shuffle[i*batch_size:(i+1)*batch_size]
        x_batch = x_train[idx]
        y_batch = y_train[idx]
        train_out = f_train(x_batch, y_batch)
        cost_train, _, train_norm = train_out[:3]

        if MONITOR:
            print(str(i) + "-"*44 + "GRAD NORM  \t UPDATE NORM \t PARAM NORM")
            all_mon = train_out[3:]
            grd_mon = train_out[:len(all_grads)]
            upd_mon = train_out[len(all_grads):]
            for pm, gm, um in zip(trainable_params, grd_mon, upd_mon):
                if '.b' not in pm.name:
                    pad = (40-len(pm.name))*" "
                    print("%s \t %.5e \t %.5e \t %.5e" % (
                        pm.name + pad,
                        np.linalg.norm(gm),
                        np.linalg.norm(um),
                        np.linalg.norm(pm.get_value())
                    ))

        cost_train_lst += [cost_train]

    conf_train = ConfusionMatrix(num_classes)
    for i in range(x_train.shape[0] // 1000):
        probs_train, _ = f_eval(x_train[i*1000:(i+1)*1000])
        preds_train_flat = probs_train.reshape((-1, num_classes)).argmax(-1)
        conf_train.batch_add(
            y_train[i*1000:(i+1)*1000].flatten(),
            preds_train_flat
        )

    if last_decay > args.decayinterval and epoch > args.nodecay:
        last_decay = 0
        old_lr = sh_lr.get_value(sh_lr)
        new_lr = old_lr / args.decayfac
        sh_lr.set_value(lasagne.utils.floatX(new_lr))
        print("Decay lr from %f to %f" % (float(old_lr), float(new_lr)))
    else:
        last_decay += 1

    # valid
    conf_valid = ConfusionMatrix(num_classes)
    for i in range(n_batches_valid):
        x_batch = x_valid[i*batch_size:(i+1)*batch_size]
        y_batch = y_valid[i*batch_size:(i+1)*batch_size]
        probs_valid, _ = f_eval(x_batch)
        preds_valid_flat = probs_valid.reshape((-1, num_classes)).argmax(-1)
        conf_valid.batch_add(
            y_batch.flatten(),
            preds_valid_flat
        )

    # test
    conf_test = ConfusionMatrix(num_classes)
    n_batches_test = x_test.shape[0] // batch_size
    all_y, all_preds = [], []
    for i in range(n_batches_test):
        x_batch = x_test[i*batch_size:(i+1)*batch_size]
        y_batch = y_test[i*batch_size:(i+1)*batch_size]
        probs_test, A_test = f_eval(x_batch)
        preds_test_flat = probs_test.reshape((-1, num_classes)).argmax(-1)
        conf_test.batch_add(
            y_batch.flatten(),
            preds_test_flat
        )

        all_y += [y_batch]
        all_preds += [probs_test.argmax(-1)]

    print("Epoch {} Acc Valid {}, Acc Train = {}, Acc Test = {}".format(
            epoch,conf_valid.accuracy(),conf_train.accuracy(),conf_test.accuracy()))

    np.savez( "res_test_3",probs=probs_test, preds=probs_test.argmax(-1),
             x=x_batch, y=y_batch, A=A_test,
             all_y=np.vstack(all_y),
             all_preds=np.vstack(all_preds))

    if conf_valid.accuracy() > best_valid:
        best_valid = conf_valid.accuracy()
        look_count = LOOK_AHEAD
    else:
        look_count -= 1

    if look_count <= 0:
        break
