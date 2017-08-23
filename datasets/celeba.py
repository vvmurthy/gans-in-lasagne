# import skimage.transform as tns
import numpy as np
import scipy.misc as misc
# "Dataset" type files are specific to a given class
# This dataset loaders are celeba specific
# They initialize the dataset, and load image files in smaller batches from hard disk


def preprocess_im(filename, li):

    # Load image
    im = misc.imread(filename)
    im.astype(np.float32)

    # Crop to [40:178, 30:168, 3]
    side_length = 138
    im = im[40: 40 + side_length, 30: 30 + side_length, :]

    # resize to li x li x 3
    if not li == 138:
        im = misc.imresize(im, (li, li, 3), interp='bilinear')

    # Swap to n_channels x image_size x image_size
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Scale to range [-1, 1]
    im = (im / np.float32(255) - np.float32(0.5)) * np.float32(2)

    return im


# Loads files after minibatch iteration
def load_files(X_files, num_samples, li):
    X = np.zeros((num_samples, 3, li, li)).astype(np.float32)
    for im in range(0, num_samples):
        X[im, :, :, :] = preprocess_im(X_files[im], li)
    return X


def load_xy(celeba_dir):

    # Creates labels
    with open(celeba_dir + 'list_attr_celeba.txt') as f:
        content = f.readlines()
    labels = content[1]
    labels = labels.split(" ")

    y = np.loadtxt(celeba_dir + 'list_attr_celeba.txt', dtype=np.float32, skiprows=2, usecols=np.arange(1, 41))

    # Keep only 18 labels used
    used_lab = (4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35)

    lab = [labels[n] for n in used_lab]
    y = y[:, used_lab]

    X_files_load = np.loadtxt(celeba_dir + 'list_attr_celeba.txt', dtype=str, skiprows=2, usecols=0)
    X_files = np.empty(X_files_load.shape[0], dtype='<S256')
    for n in range(0, X_files_load.shape[0]):
        X_files[n] = celeba_dir + 'img_align_celeba/' + X_files_load[n]

    X_files_train, y_train, X_files_val, y_val, X_files_test, y_test = train_test_split(X_files, y, celeba_dir)
    return X_files_train, y_train, X_files_val, y_val, X_files_test, y_test, lab


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
