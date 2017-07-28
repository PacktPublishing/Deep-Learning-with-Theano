from __future__ import print_function
import numpy
from theano import theano
import theano.tensor as T
import pickle, gzip
import timeit

data_dir = "/sharedfiles/"

# random generator
random_seed=1234
rng = numpy.random.RandomState(random_seed)
srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

print("Using device", theano.config.device)

print("Loading data")
with gzip.open(data_dir + "mnist.pkl.gz", 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

train_set_x = theano.shared(numpy.asarray(train_set[0], dtype=theano.config.floatX))
train_set_y = theano.shared(numpy.asarray(train_set[1], dtype='int32'))

print("Building model")

batch_size = 600
n_in=28 * 28
n_hidden=500
n_out=10

#probability to drop
dropout=0.5

x = T.matrix('x')
y = T.ivector('y')

layer0_input = x.reshape((batch_size, 1, 28, 28))

def shared_zeros(shape, dtype=theano.config.floatX, name='', n=None):
    shape = shape if n is None else (n,) + shape
    return theano.shared(numpy.zeros(shape, dtype=dtype), name=name)

from theano.tensor.nnet import conv2d

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

n_conv1 = 20

W1 = shared_glorot_uniform( (n_conv1, 1, 5, 5) )

conv1_out = conv2d(
    input=layer0_input,
    filters=W1,
    filter_shape=(n_conv1, 1, 5, 5),
    input_shape=(batch_size, 1, 28, 28)
)


from theano.tensor.signal import pool
pooled_out = pool.pool_2d(input=conv1_out, ws=(2, 2),ignore_border=True)

n_conv2 = 50

W2 = shared_glorot_uniform( (n_conv2, n_conv1, 5, 5) )

conv2_out = conv2d(
    input=pooled_out,
    filters=W2,
    filter_shape=(n_conv2, n_conv1, 5, 5),
    input_shape=(batch_size, n_conv1, 12, 12)
)

pooled2_out = pool.pool_2d(input=conv2_out, ws=(2, 2),ignore_border=True)

hidden_input = pooled2_out.flatten(2)

if dropout > 0 :
    mask = srng.binomial(n=1, p=1-dropout, size=hidden_input.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    hidden_input = hidden_input * T.cast(mask, theano.config.floatX)

n_hidden = 500

W3 = shared_glorot_uniform( (n_conv2 * 4 * 4, n_hidden), name='W3' )
b3 = shared_zeros( (n_hidden,), name='b3' )

hidden_output = T.tanh(T.dot(hidden_input, W3) + b3)

n_out = 10

if dropout > 0 :
    mask = srng.binomial(n=1, p=1-dropout, size=hidden_output.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    hidden_output = hidden_output * T.cast(mask, theano.config.floatX)

W4 = shared_zeros( (n_hidden, n_out), name='W4' )
b4 = shared_zeros( (n_out,), name='b4' )

model = T.nnet.softmax(T.dot(hidden_output, W4) + b4)
params = [W1,W2,W3,b3,W4,b4]

y_pred = T.argmax(model, axis=1)
error = T.mean(T.neq(y_pred, y))

cost = -T.mean(T.log(model)[T.arange(y.shape[0]), y])
g_params = T.grad(cost=cost, wrt=params)

learning_rate=0.1
updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(params, g_params)
    ]

index = T.lscalar()

train_model = theano.function(
    inputs=[index],
    outputs=[cost,error],
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    inputs=[x,y],
    outputs=[cost,error]
)


print("Training")

n_epochs = 1000
n_train_batches = train_set[0].shape[0] // batch_size

n_iters = n_epochs * n_train_batches
train_loss = numpy.zeros(n_iters)
train_error = numpy.zeros(n_iters)

validation_interval = 1000
n_valid_batches = valid_set[0].shape[0] // batch_size
valid_loss = numpy.zeros(n_iters / validation_interval)
valid_error = numpy.zeros(n_iters / validation_interval)

start_time = timeit.default_timer()
for epoch in range(n_epochs):
    for minibatch_index in range(n_train_batches):
        iteration = minibatch_index + n_train_batches * epoch
        train_loss[iteration], train_error[iteration] = train_model(minibatch_index)

        if iteration % validation_interval == 0 :
            val_iteration = iteration // validation_interval
            valid_loss[val_iteration], valid_error[val_iteration] = numpy.mean([
                    validate_model(
                        valid_set[0][i * batch_size: (i + 1) * batch_size],
                        numpy.asarray(valid_set[1][i * batch_size: (i + 1) * batch_size], dtype="int32")
                        )
                        for i in range(n_valid_batches)
                     ],axis=0)

            print('epoch {}, minibatch {}/{}, validation error {:02.2f} %, validation loss {}'.format(
                epoch,
                minibatch_index + 1,
                n_train_batches,
                valid_error[val_iteration] * 100,
                valid_loss[val_iteration]
            ))

end_time = timeit.default_timer()
print( end_time -start_time )

numpy.save("cnn_train_loss", train_loss)
numpy.save("cnn_valid_loss", valid_loss)
