# import skimage.transform as tns
import numpy as np
import scipy.misc as misc
import sys
import os
import sklearn.preprocessing as prep
# "Dataset" type files are specific to a given class
# This dataset loader is MNIST specific
# It will download MNIST if needed and preprocess images to 32 x 32
# Because of the small size of MNIST as compared to other datasets, we
# Do not support batch loading from memory


# We scale MNIST images to li x li by adding two rows of -1 to all 4 sides
def preprocess_im(im_, li):

    # resize to 32 x 32
    im = np.zeros((32,32), dtype=np.float32)
    im[2:30, 2:30] = im_

    # resize to li x li
    if li > 32:
        im = misc.imresize(im, (li, li), interp='bilinear')

    # Scale to range [-1, 1] from [0, 255]
    im = (im / np.float32(255) - np.float32(0.5)) * np.float32(2)

    # expand dimensions to 1 x li x li
    im = np.expand_dims(im, axis=0)

    return im


# We need this function to ensure that MNIST has a dataset loader
# Instead of reading + preprocessing the files, this function simply processes the
# preloaded images
def load_files(X_files, num_samples, li):
    X = np.zeros((num_samples, 1, li, li)).astype(np.float32)
    for im in range(0, num_samples):
        X[im, :, :, :] = preprocess_im(X_files[im], li)
    return X


# This downloads MNIST if not already and loads raw files
def load_xy(mnist_dir):
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, mnist_dir, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, mnist_dir + filename)

    import gzip

    def load_mnist_images(mnist_dir, filename):
        if not os.path.exists(mnist_dir + filename):
            download(filename, mnist_dir)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        data = data.reshape(-1, 1, 28, 28)
        return data

    def load_mnist_labels(mnist_dir, filename):
        if not os.path.exists(mnist_dir + filename):
            download(filename, mnist_dir)

        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)

        return data.astype(np.float32)

    X_train = load_mnist_images(mnist_dir, 'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(mnist_dir, 'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(mnist_dir, 't10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(mnist_dir, 't10k-labels-idx1-ubyte.gz')

    # We also return the labels of the dataset
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # We store mnist labels where -1 is negative, 1 is positive
    # To match range of tanh
    y_train = prep.label_binarize(y_train, labels)
    y_train[y_train == 0] = -1
    y_test = prep.label_binarize(y_test, labels)
    y_test[y_test == 0] = -1

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]



    return X_train, y_train, X_val, y_val, X_test, y_test, labels
