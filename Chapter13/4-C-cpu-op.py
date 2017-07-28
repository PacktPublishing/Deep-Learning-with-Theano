from __future__ import print_function
import numpy
import theano
from theano import gof


class AXPBOp(gof.Op):
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

    def c_code_cache_version(self):
        return (6, 1)

    def c_support_code(self):
        c_support_code = """
        bool same_shape(PyArrayObject* arr1, PyArrayObject* arr2)
        {
            if( PyArray_NDIM(arr1) != PyArray_NDIM(arr2)) {
                return false;
            }
            for(int i = 0; i < PyArray_NDIM(arr2) ; i++) {
                if (PyArray_DIMS(arr1)[0] == PyArray_DIMS(arr2)[0]) {
                    return false;
                }
            }
            return true;
        }
        """

        return c_support_code

    def c_support_code_apply(self, node, name):
        dtype_x = node.inputs[0].dtype
        dtype_z = node.outputs[0].dtype

        a = self.a
        b = self.b

        c_support_code = """
        void elemwise_op_%(name)s(npy_%(dtype_x)s* x_ptr, npy_intp* x_str, int itemsize_x,
            npy_%(dtype_z)s* z_ptr, npy_intp* z_str, int itemsize_z,
            int nbDims, npy_intp* dims)
        {
            npy_intp stride_x = (npy_intp)(1);
            npy_intp stride_z = (npy_intp)(1);
            for (int i = 0; i < nbDims; i ++) {
                stride_x = stride_x * x_str[i] / itemsize_x;
                stride_z = stride_z * z_str[i] / itemsize_z;
            }
            for (int i=0; i < dims[0]; i++)
                if (nbDims==1) {
                    z_ptr[i * z_str[0]/itemsize_z] = x_ptr[i * x_str[0] / itemsize_x] * ((npy_%(dtype_z)s) %(a)s) + ((npy_%(dtype_z)s)%(b)s);
                } else {
                    elemwise_op_%(name)s( x_ptr + i * stride_x , x_str + 1, itemsize_x,
                        z_ptr + i * stride_z , z_str + 1, itemsize_z,
                        nbDims - 1, dims + 1 );
                }
        }
        """

        return c_support_code % locals()

    def c_code(self, node, name, inp, out, sub):
        x = inp[0]
        z = out[0]

        dtype_x = node.inputs[0].dtype
        dtype_z = node.outputs[0].dtype

        itemsize_x = numpy.dtype(dtype_x).itemsize
        itemsize_z = numpy.dtype(dtype_z).itemsize

        typenum_z = numpy.dtype(dtype_z).num

        fail = sub['fail']

        c_code = """
        // Validate that the output storage exists and has the same
        // dimension as x.
        if (NULL == %(z)s || !(same_shape(%(x)s, %(z)s)))
        {
            /* Reference received to invalid output variable.
            Decrease received reference's ref count and allocate new
            output variable */
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*)PyArray_EMPTY(PyArray_NDIM(%(x)s),
                                                PyArray_DIMS(%(x)s),
                                                %(typenum_z)s,
                                                0);

            if (!%(z)s) {
                %(fail)s;
            }
        }

        // Perform the elemwise operation
        ((npy_%(dtype_z)s *)PyArray_DATA(%(z)s))[0] = 0;
        elemwise_op_%(name)s((npy_%(dtype_x)s*)PyArray_DATA(%(x)s), PyArray_STRIDES(%(x)s), %(itemsize_x)s,
                                (npy_%(dtype_z)s*)PyArray_DATA(%(z)s), PyArray_STRIDES(%(z)s), %(itemsize_z)s,
                                PyArray_NDIM(%(x)s), PyArray_DIMS(%(x)s) );

        """

        return c_code % locals()



mult4plus5op = AXPBOp(4,5)


x = theano.tensor.matrix()
y = mult4plus5op( x )
print(y)
theano.printing.pprint(y)
theano.printing.debugprint(y)
theano.printing.pydotprint(y)

print("Compiling")
f = theano.function([x], y)
theano.printing.debugprint(f)

print("Eval")
ind = numpy.random.rand(3,2).astype('float32')
print("Equality", numpy.allclose(f(ind), 4 * ind + 5 ))
print(mult4plus5op)
