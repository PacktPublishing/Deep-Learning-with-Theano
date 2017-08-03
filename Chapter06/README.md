# Chapter 6: locating with Spatial Transformer networks

Install Lasagne:

    pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

Download the data:

    wget http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz -P /sharedfiles

Train the MNIST model:

    python 1-train-mnist.py

Create cluttered images containing 1 digit:

    python create_mnist_sequence.py --nb_digits=1

Plot the first 3 samples:

    python plot_data.py mnist_sequence1_sample_8distortions_9x9.npz

Train the STN network to learn to localize the digit:

    python 2-stn-cnn-mnist.py

Plot the results:

    python plot_crops.py res_test_2.npz

Create cluttered images containing sequences of 3 digits:

    python create_mnist_sequence.py --nb_digits=3 --output_dim=100

Plot the first 3 images:

    python plot_data.py mnist_sequence3_sample_8distortions_9x9.npz

Train the recurrent model:

    python 3-recurrent-stn-mnist.py

Plot the results:

    python plot_crops.py res_test_3.npz
