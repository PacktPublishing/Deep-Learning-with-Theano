from __future__ import print_function
import matplotlib.pyplot as plt
import numpy

def average_per_epoch(a,epoch=30):
    n_train = len(a) / 30
    ret = numpy.cumsum(a, dtype=float)[range(0,len(a),n_train)]
    return (ret[1:] - ret[:-1])/n_train

line_style=['-', '--', '^', '-.' , ':' , 'steps']
scale = 1
h = 500
i = 0
for typ in ["simple","lstm","gru"] :
# typ = "simple"
# for h in [50,100,200,300,400,500] :
    data = numpy.load("train_loss_word_" + typ + "_h" + str(h) + "_e30.npy")
    print("Data length", len(data))
    data = data[:len(data)//scale]
    data = average_per_epoch(data)
    plt.plot(numpy.linspace(0,data.shape[0],data.shape[0]), data, line_style[i], label=typ + " " + str(h))
    i = i + 1
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
# plt.show()
plt.savefig('myfig')
