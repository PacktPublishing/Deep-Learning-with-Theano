from __future__ import print_function
import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
print(matplotlib.matplotlib_fname())
import sys, math

data = np.load(sys.argv[1])

train = data['X_train']
y_train = data['y_train']
valid = data['X_valid']
test = data['X_test']

print("Train size", train.shape)
print("Valid size", valid.shape)
print("Test size ", test.shape)

width, height = int(math.sqrt(train.shape[1])), int(math.sqrt(train.shape[1]))

for i in range(10):
    img = train[i].reshape((height, width))
    print("Label", y_train[i])
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.title("Train " + str(i), fontsize=20)
    plt.show()
