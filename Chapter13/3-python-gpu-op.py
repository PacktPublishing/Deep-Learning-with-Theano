from __future__ import print_function
import theano, numpy
from theano import Op, Apply, config, tensor
from theano.gpuarray.basic_ops import infer_context_name, as_gpuarray_variable
from theano.gpuarray.type import (GpuArrayType, GpuArrayConstant, gpu_context_type,
                   get_context, ContextNotDefined)
try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass


class GpuAXPBOp(theano.Op):
    """
    This creates an Op that takes x to a*x+b.
    """
    __props__ = ("a", "b", "context_name")

    def __init__(self, a, b, context_name=None):
        self.context_name = context_name
        self.a = a
        self.b = b
        super(GpuAXPBOp, self).__init__()

    def make_node(self, x):
        x = as_gpuarray_variable(x, self.context_name)

        x_arg = pygpu.elemwise.arg('x', 'float32', read=True)
        c_arg = pygpu.elemwise.arg('c', 'float32', read=True, write=True)
        self.my_op = pygpu.elemwise.GpuElemwise(get_context(self.context_name), "c = " + str(self.a) + " * x + " + str(self.b), [x_arg, c_arg], convert_f16=True)

        return Apply(self, [x], [x.type()])


    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = pygpu.empty(x.shape, dtype=x.dtype, context=get_context(self.context_name))
        self.my_op( x, z[0])


    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        return [a * output_grads[0]]

mult4plus5op = GpuAXPBOp(4,5)


x = theano.tensor.matrix()
y = mult4plus5op( x * 2) + 4
print(y)
theano.printing.pprint(y)
theano.printing.debugprint(y)
theano.printing.pydotprint(y)

print("Compiling")
f = theano.function([x], y)
theano.printing.debugprint(f)

print("Eval")
ind = numpy.random.rand(3,2).astype('float32')
print("Equality", numpy.allclose(f(ind),  (4 * (2 * ind) + 5 ) + 4))

print(mult4plus5op)
