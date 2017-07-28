import matplotlib.pyplot as plt
import numpy

curves = {}
curves[0] = { 'data' : numpy.load("simple_valid_loss.npy"), 'name' : "simple"}
curves[1] = { 'data' : numpy.load("mlp_valid_loss.npy"), 'name' : "mlp"}
curves[2] = { 'data' : numpy.load("conv_valid_loss.npy"), 'name' : "cnn"}

line_style=['-', '--', '^'] 
# scale = 15
# train_loss = train_loss[:len(train_loss)/scale]
# valid_loss = valid_loss[:len(valid_loss)/scale]

for c in curves :
    plt.plot(numpy.linspace(0,100,curves[c]['data'].shape[0]), curves[c]['data'], line_style[c], label=curves[c]['name'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
# plt.show()
plt.savefig('myfig')
