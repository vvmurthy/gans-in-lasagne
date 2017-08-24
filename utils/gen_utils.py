# gen_utils - utility functions used generally (by various model, training, graphing functions)
import numpy as np


# Calculates flattened dimension for convolutional layer to fully connected layer
def product(tuple_dims):
    prod = 1
    for dim in tuple_dims:
        prod = prod*dim
    return prod


# Converts an image to [0, 1] range with n x n x c dimensions (for image saving)
def deprocess_image(image, li, nc):
    image = (image / np.float32(2) + np.float32(0.5))
    im = np.zeros((li, li, nc)).astype(np.float32)
    for chan in range(0, nc):
        im[:, :, chan] = image[chan, :, :]
    return im


# Interpolate between two different vectors
def interpolate_vector(vect_1, vect_2, n_interpolations):
    weights = np.arange(0, 1 + np.float32(1 / float(n_interpolations)), 
                        np.float32(1 / float(n_interpolations)))
    vect = np.zeros((weights.shape[0], vect_1.shape[0])).astype(np.float32)

    for n in range(0, weights.shape[0]):
        for hid in range(0, vect_1.shape[0]):
            vect[n, hid] = vect_1[hid] * weights[n] + vect_2[hid] * (1 - weights[n])

    return vect