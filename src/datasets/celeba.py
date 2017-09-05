import numpy as np
import scipy.misc as misc
import os


class CelebA:
    """
        datasets.CelebA(celeba_dir= os.getcwd + '/celeba/', **kwargs)
        CelebA dataset loading utilities.
        Initializes CelebA dataset (images + 18 labels from [2]),
        and provides the following required functions:
        * ``preprocess_im``
        * ``load_files``
        *``modify_y``
        *``randomize_y``

        Parameters
        ----------
        celeba_dir : a string
            The directory where the CelebA images are stored. ``celeba_dir`` is assumed
            to have the following structure:
            + ``celeba_dir``
                + ``list_attr_celeba.txt``
                + ``img_align_celeba`` (Directory)
                + ``list_eval_partition.txt``

            This string is assumed to be an absolute path.
        **kwargs
            Any additional keywords that override a default attribute (below).
        Functions
        ---------
        load_files : Function
            Load images to memory and preprocess images into nc x li x li format and [-1, 1] range.
        modify_y : Function
            Generated ``lab_ln + 1`` number of permutations of a given label vector.
            Allows set of images with same z vector but different y vectors to be
            created. Follows constraints of labels in dataset (only 1 hair color, only 1 hair texture)
        randomize_y : Function
            Returns a random y vector. Generates permutations of same y vector from `modify_y` then
            selects a random vector to return.
        Attributes
        ----------
        nc : int
            The number of channels images in CelebA have. This is locked at 3.
        lab_ln : int
            The number of labels in CelebA. this is locked at 18.
        li : int
            The length of an input image. By default, this is 64 pix x 64 pix.
            Images are cropped and resized down to 64 x 64.
        images_in_mem : int
            The amount of images to read into memory from the the hard drive. By
            default, this is set at 12,000.
            of the dataset (50000).
        X_files_train : 2D :class:`NdArray`
            Contains absolute file paths for each image in CelebA train partition.
        y_train : 2D :class:`NdArray`
            Contains binarized labels for CelebA train partition.
        X_files_val : 4D :class:`NdArray`
            Contains absolute file paths for each image in CelebA validation partition.
        y_val : 2D :class:`NdArray`
            Contains binarized labels for CelebA validation partition.
         X_files_test : 4D :class:`NdArray`
            Contains absolute file paths for each image in CelebA test partition.
        y_test : 2D :class:`NdArray`
            Contains binarized labels for CelebA test partition.
        labels : :class:`List` Object
            Label names for MNIST. Contains descriptors such as ``['bald', 'heavy_makeup' ...
            'blonde']``
        References
        ----------
        .. [1] Ziwei Liu, Ping Luo, Xiaogang Wang, Xiaoou Tang (2015):
            Deep Learning Face Attributes in the Wild. arXiv.
            http://arxiv.org/abs/1411.7766
            http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
        .. [2] Guim Perarnau, Joost van de Weijer, Bogdan Raducanu,
            Jose M. Alvarez(2016):
            A guide to convolution arithmetic for deep learning. arXiv.
            https://arxiv.org/abs/1611.06355
        """

    def __init__(self, celeba_dir=os.getcwd() + '/celeba/', **kwargs):
        self.dir = celeba_dir

        # sets unchangeable params
        self.nc = 3
        self.lab_ln = 18

        # sets changeable params
        if 'li' not in kwargs:
            self.li = 64
        else:
            self.li = kwargs['li']

        if 'images_in_mem' not in kwargs:
            self.images_in_mem = 12000
        else:
            self.images_in_mem = kwargs['images_in_mem']

        # checks changeable params + formal args
        if(self.li < 1):
            raise ValueError(
                "Length of an image must be > 1"
            )

        if not isinstance(self.li, int):
            raise ValueError(
                "Length of Image must be integer"
            )

        if not os.path.isdir(self.dir):
            raise ImportError(
                "Directory %s Not Found" %self.dir
            )

        for fl in ['list_eval_partition.txt', 'list_attr_celeba.txt']:
            if not os.path.isfile(self.dir + fl):
                raise ImportError(
                    'File %s not found in %s' %(fl, self.dir)
                )

        if not os.path.isdir(self.dir + 'img_align_celeba'):
            raise ImportError(
                "Images not located in %s" %(self.dir + 'img_align_celeba')
            )

        # Initialize images in dataset
        self.X_files_train, \
        self.y_train, \
        self.X_files_val, \
        self.y_val, \
        self.X_files_test, \
        self.y_test, \
        self.labels = self.init_dataset()

    def preprocess_im(self, filename):

        # Load image
        im = misc.imread(filename)
        im.astype(np.float32)

        # Crop to [40:178, 30:168, 3]
        side_length = 138
        im = im[40: 40 + side_length, 30: 30 + side_length, :]

        # resize to li x li x 3
        if not self.li == 138:
            im = misc.imresize(im, (self.li, self.li, self.nc), interp='bilinear')

        # Swap to n_channels x image_size x image_size
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

        # Scale to range [-1, 1]
        im = (im / np.float32(255) - np.float32(0.5)) * np.float32(2)

        return im

    # Loads files after minibatch iteration
    def load_files(self, X_files, num_samples):
        X = np.zeros((num_samples, self.nc, self.li, self.li)).astype(np.float32)
        if num_samples == 1:
            return np.expand_dims(self.preprocess_im(X_files), axis=0)
        else:
            for im in range(0, num_samples):
                X[im, :, :, :] = self.preprocess_im(X_files[im])
        return X

    # Generates 18 permutations of y
    # y is a vector here
    def modify_y(self, y, binarize):

        # Binarize y vector given certain threshold
        def binarize_y(y, threshhold):

            type_hair = [0, 15, 16]
            color_hair = [2, 3, 4, 7]

            hair_type_index = type_hair[np.argmax(y[type_hair])]
            color_type_index = color_hair[np.argmax(y[color_hair])]

            y[type_hair] = -1
            y[hair_type_index] = 1

            y[color_hair] = -1
            y[color_type_index] = 1

            y[y <= threshhold] = -1
            y[y > threshhold] = 1
            return y

        # Binarize y
        if binarize:
            y = binarize_y(y, 0)

        type_hair = [0, 15, 16]
        color_hair = [2, 3, 4, 7]
        y_vars = np.zeros((19, 18)).astype(np.float32)

        y_vars[0, :] = y

        for n in range(0, y.shape[0]):
            temp = np.copy(y)

            if n in type_hair:
                temp[type_hair] = -1
                temp[n] = 1
            elif n in color_hair:
                temp[color_hair] = -1
                temp[n] = 1
            else:
                temp[n] = 1
            y_vars[n + 1, :] = temp

        return y_vars

    # Generates fake y vectors for a given minibatch
    def randomize_y(self, y):
        random_y = np.zeros((y.shape[0], y.shape[1], 1))
        for n in range(0, y.shape[0]):
            index = np.random.randint(0, y.shape[1])
            random_y[n, :, 0] = self.modify_y(y[n, :, 0], False)[index, :]

        return random_y.astype(np.float32)

    def init_dataset(self):

        def train_test_split(X_files, y, celeba_dir):
            split = np.loadtxt(celeba_dir + 'list_eval_partition.txt', dtype=np.float32, skiprows=0, usecols=1)
            first_val = 0
            first_test = 0
            for n in range(0, split.shape[0]):
                if first_val == 0 and split[n] == 1:
                    first_val = n
                elif first_test == 0 and split[n] == 2:
                    first_test = n

            X_files_train = X_files[0:first_val]
            y_train = y[0:first_val]

            X_files_val = X_files[first_val:first_test]
            y_val = y[first_val:first_test]

            X_files_test = X_files[first_test:]
            y_test = y[first_test:]

            return X_files_train, y_train, X_files_val, y_val, X_files_test, y_test

        # Creates labels
        with open(self.dir + 'list_attr_celeba.txt') as f:
            content = f.readlines()
        labels = content[1]
        labels = labels.split(" ")

        y = np.loadtxt(self.dir + 'list_attr_celeba.txt', dtype=np.float32, skiprows=2, usecols=np.arange(1, 41))

        # Keep only 18 labels used
        used_lab = (4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35)

        lab = [labels[n] for n in used_lab]
        y = y[:, used_lab]

        X_files_load = np.loadtxt(self.dir + 'list_attr_celeba.txt', dtype=str, skiprows=2, usecols=0)
        X_files = np.empty(X_files_load.shape[0], dtype='<S256')
        for n in range(0, X_files_load.shape[0]):
            X_files[n] = self.dir + 'img_align_celeba/' + X_files_load[n]

        X_files_train, y_train, X_files_val, y_val, X_files_test, y_test = train_test_split(X_files, y, self.dir)
        return X_files_train, y_train, X_files_val, y_val, X_files_test, y_test, lab

