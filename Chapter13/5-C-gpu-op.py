from __future__ import absolute_import, print_function, division
from theano import theano, Op, Apply, config, tensor
from theano.gpuarray.basic_ops import GpuKernelBase, Kernel, as_gpuarray_variable, infer_context_name
from theano.gpuarray.type import (GpuArrayType, GpuArrayConstant, gpu_context_type,
                   get_context, ContextNotDefined)
from theano.gpuarray.fp16_help import write_w
import numpy
try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass


class GpuAXPBOp(GpuKernelBase, Op):
    """
    This creates an Op that takes x to a*x+b
    """
    __props__ = ('dtype', 'context_name', "a", "b")
    _f16_ok = True

    def __init__(self, a, b, dtype=None, context_name=None):
        self.a = a
        self.b = b
        if dtype is None:
            dtype = config.floatX
        self.dtype = dtype
        self.context_name = context_name
        super(GpuAXPBOp, self).__init__()

    def get_params(self, node):
        return get_context(self.context_name)

    def make_node(self, x):
        x = as_gpuarray_variable(x, infer_context_name(x))
        return theano.Apply(self, [x], [x.type()])

    def infer_shape(self, node, shape):
        return shape

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i])
                for i in xrange(1)]

    def gpu_kernels(self, node, name):
        code = """
KERNEL void axpb(GLOBAL_MEM %(ctype)s *x, GLOBAL_MEM  %(ctype)s *z, ga_size n, ga_size m) {
    for (ga_size i = LID_0; i < n; i += LDIM_0) {
        for (ga_size j = LID_0; j < m; j += LDIM_0) {
            z[i*m + j] = %(write_a)s( 2 * x[i*m + j] );
        }
    }
}""" % dict(ctype=pygpu.gpuarray.dtype_to_ctype(self.dtype),
            name=name, write_a=write_w(self.dtype))
        return [Kernel(
                code=code, name="axpb",
                params=[gpuarray.GpuArray, gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SIZE],
                flags=Kernel.get_flags(self.dtype),
                objvar='k_axpb_' + name)]

    def c_code(self, node, name, inp, out, sub):
        n, = inp
        z, = out
        dtype_n = node.inputs[0].dtype
        fail = sub['fail']
        ctx = sub['params']
        typecode = pygpu.gpuarray.dtype_to_typecode(self.dtype)
        sync = bool(config.gpuarray.sync)
        kname = self.gpu_kernels(node, name)[0].objvar
        s = """
        size_t dims[2] = {0, 0};
        size_t ls, gs;
        int err;
        dims[0] = %(n)s->ga.dimensions[0];
        dims[1] = %(n)s->ga.dimensions[1];
        Py_CLEAR(%(z)s);
        %(z)s = pygpu_zeros(2, dims,
                            %(typecode)s,
                            GA_C_ORDER,
                            %(ctx)s, Py_None);
        if (%(z)s == NULL) {
            %(fail)s
        }
        ls = 1;
        gs = 256;
        err = axpb_call(1, &gs, &ls, 0, %(n)s->ga.data, %(z)s->ga.data, dims[0], dims[1]);
        if (err != GA_NO_ERROR) {
            PyErr_Format(PyExc_RuntimeError,
                         "gpuarray error: kEye: %%s. n%%lu, m=%%lu.",
                         GpuKernel_error(&%(kname)s, err),
                         (unsigned long)dims[0], (unsigned long)dims[1]);
            %(fail)s;
        }
        if(%(sync)d)
            GpuArray_sync(&%(z)s->ga);
        """ % locals()

        return s

    def c_code_cache_version(self):
        return (21,4)


mult4plus5op = GpuAXPBOp(4,5)

x = theano.tensor.matrix('x')
z = mult4plus5op(x)

theano.printing.debugprint(z)
print("Compiling")
f = theano.function( [x], z)
theano.printing.debugprint(f)

print("Eval")
ind = numpy.random.rand(3,2).astype(theano.config.floatX)
print("Equality", numpy.allclose(f(ind), 2 * ind ))
print(mult4plus5op)
