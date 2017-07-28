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
n_out=10

x = T.matrix('x')
y = T.ivector('y')
W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
b = theano.shared(
    value=numpy.zeros(
        (n_out,),
        dtype=theano.config.floatX
    ),
    name='b',
    borrow=True
)
model = T.nnet.softmax(T.dot(x, W) + b)

y_pred = T.argmax(model, axis=1)
error = T.mean(T.neq(y_pred, y))

cost = -T.mean(T.log(model)[T.arange(y.shape[0]), y])
g_W = T.grad(cost=cost, wrt=W)
g_b = T.grad(cost=cost, wrt=b)

learning_rate=0.13
index = T.lscalar()

train_model = theano.function(
    inputs=[index],
    outputs=[cost,error],
    updates=[(W, W - learning_rate * g_W),(b, b - learning_rate * g_b)],
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

n_epochs = 100
n_train_batches = train_set[0].shape[0] // batch_size

n_iters = n_epochs * n_train_batches
train_loss = numpy.zeros(n_iters)
train_error = numpy.zeros(n_iters)

validation_interval = 100
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

numpy.save("simple_train_loss", train_loss)
numpy.save("simple_valid_loss", valid_loss)
