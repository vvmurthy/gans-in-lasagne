import os

import lasagne
import numpy as np
import scipy.misc as misc
import theano
import theano.tensor as T

from datasets.celeba import load_xy


def preprocess_im(filename, li):

    # Load image
    im = misc.imread(filename)

    # resize to li x li x 3
    im = misc.imresize(im, (li, li, 3), interp='bilinear')

    # Swap to n_channels x image_size x image_size
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    return im
    
def load_files(X_files, num_samples, li):
    X = np.zeros((num_samples, 3, li, li)).astype(np.float32)
    for im in range(0, num_samples):
        X[im, :, :, :] = preprocess_im(X_files[im], li)
    return X
    
    
    
def their_version(x1, scale_factor):
    indices = np.arange(0, x1*scale_factor)
    source_indices = np.zeros((indices.shape[0]))
    
    for x in indices:
        source_indices[x] = np.min(np.array([np.floor(np.float32(x) / scale_factor), x1 - 1]))
    
    dot_matrix = np.zeros((x1, indices.shape[0])).astype(np.float32)
    source_indices = source_indices.astype(np.int)
    for col in range(0, dot_matrix.shape[1]):
        dot_matrix[ source_indices[col], col] = 1
    
    return dot_matrix

X_files_train, y_train, X_files_val, y_val, X_files_test, y_test, lab = load_xy(os.getcwd() + '/')

li = 64
scale_factor= 2
x_ims = load_files(X_files_val[0:100], 100, li)
dot_matrix= their_version(li, scale_factor)
inputs = T.tensor4('inputs')
l_in = lasagne.layers.InputLayer((None, 3, li, li), input_var=inputs)
l_1 = lasagne.layers.DenseLayer(l_in, li*scale_factor, W=dot_matrix, b=None, nonlinearity=None, num_leading_axes=-1)
l_2 = lasagne.layers.DimshuffleLayer(l_1, (0, 1, 3, 2))
l_3 = lasagne.layers.DenseLayer(l_2, li*scale_factor, W=dot_matrix, b=None, nonlinearity=None, num_leading_axes=-1)
l_4 = lasagne.layers.DimshuffleLayer(l_3, (0, 1, 3, 2))
nearest = theano.function([inputs], lasagne.layers.get_output(l_4))

li2 = 3
scale_factor_2= 3
x_ims_2 = load_files(X_files_test[0:100], 100, li2)
dot_matrix_2= their_version(li2, scale_factor_2)
inputs = T.tensor4('inputs')
l_in = lasagne.layers.InputLayer((None, 3, li2, li2), input_var=inputs)
l_1 = lasagne.layers.DenseLayer(l_in, li2*scale_factor_2, W=dot_matrix_2, b=None, nonlinearity=None, num_leading_axes=-1)
l_2 = lasagne.layers.DimshuffleLayer(l_1, (0, 1, 3, 2))
l_3 = lasagne.layers.DenseLayer(l_2, li2*scale_factor_2, W=dot_matrix_2, b=None, nonlinearity=None, num_leading_axes=-1)
l_4 = lasagne.layers.DimshuffleLayer(l_3, (0, 1, 3, 2))
nearest_2 = theano.function([inputs], lasagne.layers.get_output(l_4))
    

#### Checking Nearest Neighbor Upscale Implementation ######
# Implementation is designed to work with square images, integer scale factor of 2
# For use with BEGAN Generative Net
# Validates against scipy imresize 'nearest' mode

# Load images for validation

x_scipy = np.zeros((x_ims.shape[0], 3, li*scale_factor, li*scale_factor)).astype(np.float32)
x_scipy_2 = np.zeros((x_ims_2.shape[0], 3, li2*scale_factor_2, li2*scale_factor_2)).astype(np.float32)

for n in range(0, x_scipy.shape[0]):
    intermediate = misc.imresize(x_ims[n], (x_scipy.shape[2], x_scipy.shape[3], 3), interp='nearest') 
    for chan in range(0, 3):
        x_scipy[n, chan, :, :] = intermediate[:, :, chan]
    intermediate = misc.imresize(x_ims_2[n], (x_scipy_2.shape[2], x_scipy_2.shape[3], 3), interp='nearest') 
    for chan in range(0, 3):
        x_scipy_2[n, chan, :, :] = intermediate[:, :, chan]
        
x_res = nearest(x_ims)
x_res_2 = nearest_2(x_ims_2)

# See if the two results match
print(np.allclose(x_res, x_scipy))
print(np.allclose(x_res_2, x_scipy_2))
