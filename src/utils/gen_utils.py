# gen_utils - utility functions used generally (by various model, training, graphing functions)
import numpy as np


def product(tuple_dims):
    """
    utils.product(tuple_dims)
    Calculates flattened dimension for convolutional layer to fully connected layer.
    Used as a helper function to display parameters between conv and fc connection.

    Parameters
    ----------
    tuple_dims : tuple or list
        Dimensions which to be multipled.
    Returns
    -------
    prod : int
        The product of input dimensions.
    Examples
    --------

    >>> dims = [3, 10, 10]
    >>> print prod(dims)

    300
    """
    prod = 1
    for dim in tuple_dims:
        prod = prod*dim
    return prod


def deprocess_image(image, li, nc):
    """
    utils.deprocess_image(image, li, nc)
    Converts an image to [0, 1] range with n x n x c dimensions. Scipy's ``misc``
    utilities require a float32-type image to be in range [0, 1] with ``li x li x nc``
    dimensions. This function converts an image from [-1, 1], ``nc x li x li`` to scipy
    misc format.

    Parameters
    ----------
    image : 3D :class:``NdArray``
        Image in Tanh domain, with dimension format ``nc x li x li``.
    li : int
        Length of the image to be converted
    nc : int
        Number of channels in image to be converted
    Returns
    -------
    im : :class:``NdArray``
        Image in [0, 1] domain with dimension format ``li x li x nc``
    """
    image = (image / np.float32(2) + np.float32(0.5))
    im = np.zeros((li, li, nc)).astype(np.float32)
    for chan in range(0, nc):
        im[:, :, chan] = image[chan, :, :]
    return im


def interpolate_vector(vect_1, vect_2, n_interpolations):
    """
    utils.interpolate_vector(vect1, vect2, n_interpolations)
    Interpolates between two vectors to create a matrix of interpolated vectors.

    Parameters
    ----------
    vect_1 : 1D :class:``NdArray``
        Image in Tanh domain, with dimension format ``nc x li x li``.
    vect_2 : 1D :class:``NdArray``
        Length of the image to be converted
    n_interpolations : int
        Number of steps to take between interpolations. The total number of
        images in ``n_interpolatons`` steps is ``n_interpolations + 2``
        (intermediate + two endpoints)
    Returns
    -------
    vect : 2D :class:``NdArray``
        Set of interpolated vectors with dimensions ``[n_interpolations + 2, vect_1.shape]``
    """
    weights = np.arange(0, 1 + np.float32(1 / float(n_interpolations)),
                        np.float32(1 / float(n_interpolations)))
    vect = np.zeros((weights.shape[0], vect_1.shape[0])).astype(np.float32)

    for n in range(0, weights.shape[0]):
        for hid in range(0, vect_1.shape[0]):
            vect[n, hid] = vect_1[hid] * weights[n] + vect_2[hid] * (1 - weights[n])

    return vect