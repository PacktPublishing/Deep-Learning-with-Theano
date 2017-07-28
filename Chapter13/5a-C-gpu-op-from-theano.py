import theano
from theano.gpuarray.basic_ops import GpuEye
x = theano.tensor.iscalar('x')
y = theano.tensor.iscalar('y')
z = GpuEye(dtype='float32', context_name=None)(x,y, theano.tensor.constant(0))
theano.printing.debugprint(z)
print("Compiling")
f = theano.function( [x,y], z)
theano.printing.debugprint(f)
print("Results")
print(f(3, 3))
