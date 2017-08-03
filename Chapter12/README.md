# Chapter 12: learning features with Unsupervised Generative Networks

Train the model composed of 1 RBM with the contrastive divergence algorithm on MNIST dataset:

    python 1-rbm.py

Install the progress bar tqdm and Scikit-learn packages:

    conda install --file requirements.txt

to train the Deep Convolutional GAN network on MNIST:

    python 2-train_dcgan.py

**Erratum**: please not the usage of `GpuDnnConvDesc` and `GpuAllocEmpty` when using the new GPU backend (the one used in the book):

```python
desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, conv_mode=conv_mode)(kerns.shape)
out = GpuAllocEmpty(dtype='float32', context_name=infer_context_name(X))(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
```
