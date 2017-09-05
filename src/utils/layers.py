import numpy as np
import lasagne


# two convolutions and a batch normalization
def batch_conv(network, prev_name, name, num_filters, filter_size, stride, pad, W, nonlinearity):
    if W == False:
        network[name + '_conv'] = lasagne.layers.Conv2DLayer(network[prev_name], num_filters,
                                                             filter_size, stride=2, pad=pad, b=None, nonlinearity=None)
    else:
        network[name + '_conv'] = lasagne.layers.Conv2DLayer(network[prev_name], num_filters,
                                                             filter_size, stride=2, pad=pad, W=W, b=None,
                                                             nonlinearity=None)

    # Batch Normalization
    network[name + '_bn'] = lasagne.layers.BatchNormLayer(network[name + '_conv'])

    # Then nonlinearity
    network[name] = lasagne.layers.NonlinearityLayer(network[name + '_bn'], nonlinearity=nonlinearity)

    return network


# Simplified implementation of nearest neighbor in lasagne
def nearest_neighbor(network, name, prev_name, li, scale_factor):
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
