from __future__ import print_function
import pygpu
a_arg = pygpu.elemwise.arg('a', 'float32', read=True)
b_arg = pygpu.elemwise.arg('b', 'float32', read=True)
c_arg = pygpu.elemwise.arg('c', 'float32', read=True, write=True)
ctx = pygpu.gpuarray.GpuContext('cuda', 0, 0)
my_op = pygpu.elemwise.GpuElemwise(ctx, "c = a + b", [a_arg, b_arg, c_arg], convert_f16=True)

a = pygpu.gpuarray.empty((2,2),dtype='float32',context=ctx)
b = pygpu.gpuarray.empty((1,1),dtype='float32',context=ctx)
c = pygpu.gpuarray.empty((2,2),dtype='float32',context=ctx)
b[0,0]=3
my_op(a,b,c)
print("res", c)
