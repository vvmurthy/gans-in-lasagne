import numpy as np
import scipy.misc as misc
import sys
import os
import sklearn.preprocessing as prep


class Mnist:
    """
        datasets.Mnist(mnist_dir= os.getcwd + '/mnist/', **kwargs)
        MNIST dataset loading utilities.
        Initializes MNIST dataset (images + labels), and provides the following
        required functions:
        * ``preprocess_im``
        * ``load_files``
        *``modify_y``
        *``randomize_y``

        Parameters
        ----------
        mnist_dir : a string
            The directory where the .tgz files from MNIST are contained (or should
            be contained.) This directory is assumed to be an absolute path.
        **kwargs
            Any additional keywords that override a default attribute (below).
        Functions
        ---------
        load_files : Function
            Preprocess preloaded images into nc x li x li format and [-1, 1] range.
        modify_y : Function
            Generated ``lab_ln + 1`` number of permutations of a given label vector.
            Allows set of images with same z vector but different y vectors to be
            created
        randomize_y : Function
            Returns a random y vector.
        Attributes
        ----------
        nc : int
            The number of channels images in MNIST have. This is locked at 1.
        lab_ln : int
            The number of labels in MNIST. this is locked at 10.
        li : int
            The length of an input image. By default, this is 32 pix x 32 pix.
            Images in MNIST are 28 x 28 by default, so we pad each image with
            black pixels around the edges. if one chooses an li > 32, the image
            will be bilinear upscaled to the correct image size.
        images_in_mem : int
            The amount of images to read into memory from the the hard drive. MNIST
            does not support partial reading of the dataset into memory because
            of its limited size, so this parameter is typically locked at the size
            of the dataset (50000).
        X_files_train : 4D :class:`NdArray`
            Contains 50,000 training images from MNIST read into memory.
        y_train : 2D :class:`NdArray`
            Contains labels for MNIST. Labels have been binarized upon loading.
        X_files_val : 4D :class:`NdArray`
            Contains the last 10,000 training images from MNIST as validation images.
        y_val : 2D :class:`NdArray`
            Contains binarized labels for each validation image.
         X_files_test : 4D :class:`NdArray`
            Contains the 10,000 validation images.
        y_test : 2D :class:`NdArray`
            Contains binarized labels for each test image.
        labels : :class:`List` Object
            Label names for MNIST. Contains, ``['0', '1', ... '9']``

        """

    def __init__(self, mnist_dir=os.getcwd() + '/mnist/', **kwargs):
        self.dir = mnist_dir

        # sets unchangeable params
        self.nc = 1
        self.lab_ln = 10

        # sets changeable params
        if 'li' not in kwargs:
            self.li = 32
        else:
            self.li = kwargs['li']

        if 'images_in_mem' not in kwargs:
            self.images_in_mem = 50000
        else:
            self.images_in_mem = kwargs['images_in_mem']

        # checks changeable params + formal args
        if (self.li < 1):
            raise ValueError(
                "Length of an image must be > 1"
            )

        if not isinstance(self.li, int):
            raise ValueError(
                "Length of Image must be integer"
            )

        if not os.path.isdir(self.dir):
            raise ImportError(
                "Directory %s Not Found" % self.dir
            )

        # Initialize images in dataset
        self.X_files_train, \
        self.y_train, \
        self.X_files_val, \
        self.y_val, \
        self.X_files_test, \
        self.y_test, \
        self.labels = self.init_dataset()

    # We scale MNIST images to li x li by adding two rows of -1 to all 4 sides
    def preprocess_im(self, im_):

        # resize to 32 x 32
        im = np.zeros((32,32), dtype=np.float32)
        im[2:30, 2:30] = im_

        # resize to li x li
        if self.li > 32:
            im = misc.imresize(im, (self.li, self.li), interp='bilinear')

        # Scale to range [-1, 1] from [0, 255]
        im = (im / np.float32(255) - np.float32(0.5)) * np.float32(2)

        # expand dimensions to 1 x li x li
        im = np.expand_dims(im, axis=0)

        return im

    # We need this function to ensure that MNIST has a dataset loader
    # Instead of reading + preprocessing the files, this function simply processes the
    # preloaded images
    def load_files(self, X_files, num_samples):
        X = np.zeros((num_samples, self.nc, self.li, self.li)).astype(np.float32)
        for im in range(0, num_samples):
            X[im, :, :, :] = self.preprocess_im(X_files[im])
        return X

    # Input is a vector y, returns matrix of y variations
    # Y is assumed to be mutually exclusive here
    def modify_y(self, y, binarize):

        # Binarize y in a mutually exclusive manner
        def binarize_y(y):
            top_preds = np.argmax(y)
            y[:] = -1
            y[top_preds] = 1
            return y

        # Binarize y
        if binarize:
            y = binarize_y(y)

        y_vars = np.zeros((y.shape[0] + 1, y.shape[0])).astype(np.float32)

        y_vars[0, :] = y

        indices = np.arange(0, y.shape[0])
        for n in range(0, y.shape[0]):
            temp = np.copy(y)
            temp[indices] = -1
            temp[n] = 1
            y_vars[n + 1, :] = temp

        return y_vars

    # Generates fake y vectors for a given minibatch
    def randomize_y(self, y):
        indices = np.arange(0, y.shape[1])
        np.random.shuffle(indices)
        random_y = np.zeros((y.shape[0], y.shape[1], 1)).astype(np.float32) - 1
        for n in range(0, y.shape[0]):
            index = np.random.randint(0, y.shape[1])
            random_y[n, :, 0][indices[index]] = 1

        return random_y.astype(np.float32)

    # This downloads MNIST if not already and loads raw files
    def init_dataset(self):
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
            with gzip.open(mnist_dir + filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)

            data = data.reshape(-1, 1, 28, 28)
            return data

        def load_mnist_labels(mnist_dir, filename):
            if not os.path.exists(mnist_dir + filename):
                download(filename, mnist_dir)

            with gzip.open(mnist_dir + filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)

            return data.astype(np.float32)

        X_train = load_mnist_images(self.dir, 'train-images-idx3-ubyte.gz')
        y_train = load_mnist_labels(self.dir, 'train-labels-idx1-ubyte.gz')
        X_test = load_mnist_images(self.dir, 't10k-images-idx3-ubyte.gz')
        y_test = load_mnist_labels(self.dir, 't10k-labels-idx1-ubyte.gz')

        # We also return the labels of the dataset
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        int_lab = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.float32)

        # We store mnist labels where -1 is negative, 1 is positive
        # To match range of tanh
        y_train = prep.label_binarize(y_train, int_lab)
        y_train[y_train == 0] = -1
        y_test = prep.label_binarize(y_test, int_lab)
        y_test[y_test == 0] = -1

        # We reserve the last 10000 training examples for validation.
        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

        return X_train, y_train, X_val, y_val, X_test, y_test, labels
