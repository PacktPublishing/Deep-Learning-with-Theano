from __future__ import print_function
import numpy
from theano import theano
import theano.tensor as T
import pickle, gzip
import timeit

data_dir = "/sharedfiles/"

print("Using device", theano.config.device)

print("Loading data")
with gzip.open(data_dir + "mnist.pkl.gz", 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

train_set_x = theano.shared(numpy.asarray(train_set[0],  dtype=theano.config.floatX))
train_set_y = theano.shared(numpy.asarray(train_set[1],  dtype='int32'))

print("Building model")

batch_size = 600
n_in=28 * 28
n_hidden=500
n_out=10

x = T.matrix('x')
y = T.ivector('y')

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

W1 = shared_glorot_uniform( (n_in, n_hidden), name='W1' )
b1 = shared_zeros( (n_hidden,), name='b1' )

hidden_output = T.tanh(T.dot(x, W1) + b1)

W2 = shared_zeros( (n_hidden, n_out), name='W2' )
b2 = shared_zeros( (n_out,), name='b2' )


params = [W1,b1,W2,b2]

model = T.nnet.softmax(T.dot(hidden_output, W2) + b2)

y_pred = T.argmax(model, axis=1)
error = T.mean(T.neq(y_pred, y))

cost = -T.mean(T.log(model)[T.arange(y.shape[0]), y]) + 0.0001 * (W1 ** 2).sum() + 0.0001 * (W2 ** 2).sum()

g_params = T.grad(cost=cost, wrt=params)

learning_rate=0.01
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

numpy.save("mlp_train_loss", train_loss)
numpy.save("mlp_valid_loss", valid_loss)
