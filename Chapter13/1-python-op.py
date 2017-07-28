from __future__ import print_function
import theano, numpy

class AXPBOp(theano.Op):
    """
    This creates an Op that takes x to a*x+b.
    """
    __props__ = ("a", "b")

    def __init__(self, a, b):
        self.a = a
        self.b = b
        super(AXPBOp, self).__init__()

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = self.a * x + self.b

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        return [a * output_grads[0]]

mult4plus5op = AXPBOp(4,5)


x = theano.tensor.matrix()
y = mult4plus5op(x)
print(y)
theano.printing.pprint(y)
theano.printing.debugprint(y)
theano.printing.pydotprint(y)

print("Compiling")
f = theano.function([x], y)
theano.printing.debugprint(f)

print("Eval")
ind = numpy.random.rand(3,2)
print("Equality", numpy.allclose(f(ind),  4 * ind + 5 ))

print(mult4plus5op)
