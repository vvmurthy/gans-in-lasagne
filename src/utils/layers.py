import numpy as np
import lasagne


def batch_conv(network, prev_name, name, num_filters, filter_size, stride, pad, W, nonlinearity):
    """
    utils.batch_conv(network, prev_name, name, num_filters, filter_size, stride, pad, W, nonlinearity)
    Creates set of 2 convolutions, then performs batch normalization and applies a nonlinearity.
    Convenience function.

    Parameters
    ----------
    network : dict
        a dict of layers for the input network.
    prev_name : string
        Name of the layer (key in dict) preceding the ``batch_conv``
    name : string
        Name to assign to the final layer of ``batch_conv``.
    num_filters : int
        Number of filters to use in each convolution.
    filter_size : int
        filter size (sometimes referred to as kernel size) for the convolution
    stride : int
        Stride for the convolution.
    pad : string
        Padding to use for the convolution. Can choose from lasagne keywords {'same', 'valid', 'full'}
    W : Theano Shared Tensor, Numpy :class:`NdArray`, bool
        Whether to use predefined weights. if False, initializes weights.
    nonlinearity : :class:`Lasagne.Nonlinearity`
        Nonlinearity to apply after batch normalization. Set to None for no nonlinearity.
    Returns
    -------
    network : dict
        The input network with added convolutions, batch normalization and nonlinearity.
    """
    if W == False:
        network[name + '_conv'] = lasagne.layers.Conv2DLayer(network[prev_name], num_filters,
                                                             filter_size, stride=stride, pad=pad, b=None, nonlinearity=None)
    else:
        network[name + '_conv'] = lasagne.layers.Conv2DLayer(network[prev_name], num_filters,
                                                             filter_size, stride=stride, pad=pad, W=W, b=None,
                                                             nonlinearity=None)

    # Batch Normalization
    network[name + '_bn'] = lasagne.layers.BatchNormLayer(network[name + '_conv'])

    # Then nonlinearity
    network[name] = lasagne.layers.NonlinearityLayer(network[name + '_bn'], nonlinearity=nonlinearity)

    return network


def nearest_neighbor(network, name, prev_name, li, scale_factor):
    """
    utils.nearest_neighbor(network, name, prev_name, li, scale_factor)
    Performs nearest neighbor upscale on input image.

    Parameters
    ----------
    network : dict
        a dict of layers for the input network.
    name : string
        Name to assign to the final layer of ``batch_conv``.
    prev_name : string
        Name of the layer (key in dict) preceding the ``batch_conv``
    li : int
        Length of input images (trailing 2 axes of 4D tensor)
    scale_factor : int
        Amount which to scale each image by.
    Returns
    -------
    network : dict
        The input network with nearest neighbor upscaling. Output size
        is ``[batchsize, num_filters, li * scale_factor, li * scale_factor]``
    """
    # Creates weight matrix for nearest neighbor interpolation
    def create_weights_nn(x1, scale_factor):
        indices = np.arange(0, x1 * scale_factor)
        source_indices = np.zeros((indices.shape[0]))

        for x in indices:
            source_indices[x] = np.min(np.array([np.floor(np.float32(x) // scale_factor), x1 - 1]))

        dot_matrix = np.zeros((x1, indices.shape[0])).astype(np.float32)
        source_indices = source_indices.astype(np.int)
        for col in range(0, dot_matrix.shape[1]):
            dot_matrix[source_indices[col], col] = 1

        return dot_matrix

    dot_matrix = create_weights_nn(li, scale_factor)
    network[name + '_dense1'] = lasagne.layers.DenseLayer(network[prev_name], li * scale_factor, W=dot_matrix, b=None,
                                                          nonlinearity=None,
                                                          num_leading_axes=-1)

    # Set dense layer params as non trainable so that weights for nn do not get changed
    network[name + '_dense1'].params[network[name + '_dense1'].W].remove('trainable')

    network[name + '_dims'] = lasagne.layers.DimshuffleLayer(network[name + '_dense1'], (0, 1, 3, 2))
    network[name + '_dense2'] = lasagne.layers.DenseLayer(network[name + '_dims'], li * scale_factor, W=dot_matrix,
                                                          b=None, nonlinearity=None,
                                                          num_leading_axes=-1)

    # Set dense layer params as non trainable so that weights for nn do not get changed
    network[name + '_dense2'].params[network[name + '_dense2'].W].remove('trainable')

    network[name] = lasagne.layers.DimshuffleLayer(network[name + '_dense2'], (0, 1, 3, 2))

    return network
