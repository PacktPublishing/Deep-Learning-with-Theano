import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt

names = ["Rectified Linear Unit (ReLU)", "Leaky Rectifier Linear Unit (Leaky ReLU)",
            "Sigmoid", "Hard Sigmoid", "Hard Tanh", "Tanh"]

a = T.vector('a')

def fn(x):
    res = []
    for y in [(x + T.abs_(x)) / 2.0,
        ( 1.03 * x + 0.97 * T.abs_(x) ) / 2.0,
        T.nnet.sigmoid(x),
        T.clip(x + 0.5, 0., 1.),
        T.clip(x, -1., 1.),
        T.tanh(x)]:
        res.append( y )
        res.append( T.grad(y, x) )
    return res

results, updates = theano.scan(fn, sequences=a)

print("Compiling")
f = theano.function([a], results)

print("Display")
t = np.arange(-6, 6, 0.05).astype(theano.config.floatX)
res = f(t)
print("length", len(res))
print(res)
print(res[0].shape, res[1].shape)
for i in range(6):
    fig, ax = plt.subplots()
    plt.title(names[i])
    plt.axis([-6, 6, -1.2, 1.2])
    ax.plot( t, res[i*2], '-', label='Activation')
    ax.plot( t, res[i*2+1],  '--', label='Derivative') #, t, t**2, , t, t**3, 'g^')
    legend = ax.legend(loc='lower right')
    plt.axhline(0, color='black',  alpha=0.1)
    plt.axvline(0, color='black',  alpha=0.1)
    plt.show()
