import lasagne
import numpy as np
import theano
import theano.tensor as T

from utils.gen_utils import product


# Creates weight matrix for nearest neighbor interpolation
def create_weights_nn(x1, scale_factor):
    indices = np.arange(0, x1 * scale_factor)
    source_indices = np.zeros((indices.shape[0]))

    for x in indices:
        source_indices[x] = np.min(np.array([np.floor(np.float32(x) / scale_factor), x1 - 1]))

    dot_matrix = np.zeros((x1, indices.shape[0])).astype(np.float32)
    source_indices = source_indices.astype(np.int)
    for col in range(0, dot_matrix.shape[1]):
        dot_matrix[source_indices[col], col] = 1

    return dot_matrix


# Simplified implementation of nearest neighbor in lasagne
def nearest_neighbor(network, name, prev_name, li, scale_factor):
    dot_matrix = create_weights_nn(li, scale_factor)
    network[name + '_dense1'] = lasagne.layers.DenseLayer(network[prev_name], li * scale_factor, W=dot_matrix, b=None, nonlinearity=None,
                                    num_leading_axes=-1)

    # Set dense layer params as non trainable so that weights for nn do not get changed
    network[name + '_dense1'].params[network[name + '_dense1'].W].remove('trainable')

    network[name + '_dims'] = lasagne.layers.DimshuffleLayer(network[name + '_dense1'], (0, 1, 3, 2))
    network[name + '_dense2'] = lasagne.layers.DenseLayer(network[name + '_dims'], li * scale_factor, W=dot_matrix, b=None, nonlinearity=None,
                                    num_leading_axes=-1)

    # Set dense layer params as non trainable so that weights for nn do not get changed
    network[name + '_dense2'].params[network[name + '_dense2'].W].remove('trainable')

    network[name] = lasagne.layers.DimshuffleLayer(network[name + '_dense2'], (0, 1, 3, 2))

    return network
    
    
def make_train_fns(li, gamma_in, num_filters):

    # defines variables
    print("Building model and compiling functions...")

    # Builds discriminator and generator
    k_t = T.fscalar('k_t')
    gamma = theano.compile.shared(gamma_in)
    discriminator, input_var = build_discriminator(li, num_filters)
    generator, z_var = build_generator(li, num_filters)
    
    # Gets output image from generator, discriminator
    # as well as reconstruction of generated image from discriminator
    real_out = lasagne.layers.get_output(discriminator['out'])
    ims = lasagne.layers.get_output(generator['gen_out'], z_var)
    fake_out = lasagne.layers.get_output(discriminator['out'], ims)

    # Create loss expressions
    real_diff = T.abs_(real_out - input_var).mean()
    generator_loss = T.abs_(fake_out - ims).mean()

    discriminator_loss = real_diff - k_t * generator_loss
    k_lr = theano.compile.shared(0.001)
    k_tp = k_t + k_lr * (gamma * real_diff - generator_loss)
    convergence_measure = real_diff + T.abs_(gamma * real_diff - generator_loss)

    # Updates the paramters
    # Berthelot et al use lr of 0.0001 and halve it when convergence stalls
    # The authors also do not list beta values for adam hence we use Perarnau et al's 0.5
    generator_params = lasagne.layers.get_all_params(generator['gen_out'], trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator['out'], trainable=True)
    lr = T.fscalar('lr')
    gen_updates = lasagne.updates.adam(
        generator_loss, generator_params, learning_rate=lr, beta1=0.5)
    dis_updates = lasagne.updates.adam(discriminator_loss, discriminator_params, learning_rate=lr, beta1=0.5)

    # Compiles training function
    gen_train_fn = theano.function([z_var, lr], [fake_out, ims, generator_loss], updates=gen_updates)
    dis_train_fn = theano.function([input_var, z_var, lr, k_t], [real_out, k_tp, discriminator_loss, convergence_measure],
                                   updates=dis_updates)

    # generates images for validation
    gen_fn = theano.function([z_var], lasagne.layers.get_output(generator['gen_out'], deterministic=True))
    print("...Done")    
    return generator, discriminator, gen_train_fn, gen_fn, dis_train_fn


def build_generator(li, num_filters):

    z_var = T.fmatrix('z_var')
    generator = {}
    details = [['Layer Name', 'Dims in', 'shape of layer', 'Dims out']]

    name = 'gen_z_var'
    input_shape = (None, num_filters*8*8)
    generator[name] = lasagne.layers.InputLayer(shape=input_shape, input_var=z_var)
    output_dims = input_shape
    
    # Reshape to batchsize x num_filters x 8 x 8
    prev_name = name
    name = 'gen_reshape'
    generator[name] = lasagne.layers.ReshapeLayer(generator[prev_name], ([0], num_filters, 8, 8))
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(generator[name])
    details.append([name, str(prev_output_dims), str('reshape to 4D'),
                    str(output_dims)])

    # two convolutions + upscale in each repeat
    filter_size = 3
    scale_factor = 2

    # calculates repeats - we assume li of 2^x where x is a positive integer - (e.g. 64, 128, 32, etc)
    repeat_num = int(np.log2(np.array(li)) - 3)

    for n in range(0, repeat_num):
        prev_name = name
        name = 'gen_conv' + str(n)
        prev_num_filters = lasagne.layers.get_output_shape(generator[prev_name])[1]
        generator[name] = lasagne.layers.Conv2DLayer(generator[prev_name], num_filters,
                                                     filter_size, stride=1, pad='same',
                                                     nonlinearity=lasagne.nonlinearities.elu)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(generator[name])
        details.append([name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                        str(output_dims)])

        prev_name = name
        name = 'gen_conv' + str(n) + '_' + str(2)
        prev_num_filters = lasagne.layers.get_output_shape(generator[prev_name])[1]
        generator[name] = lasagne.layers.Conv2DLayer(generator[prev_name], num_filters,
                                                     filter_size, stride=1, pad='same',
                                                     nonlinearity=lasagne.nonlinearities.elu)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(generator[name])
        details.append([name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                        str(output_dims)])

        prev_name = name
        name = 'gen_' + str(n) + '_upscale'
        generator = nearest_neighbor(generator, name, prev_name, output_dims[2], scale_factor)
        output_dims = lasagne.layers.get_output_shape(generator[name])
        details.append([name, str(prev_output_dims), str('2x upscale'),
                        str(output_dims)])

    prev_name = name
    name = 'gen_out'
    num_filters = 3
    filter_size = 3
    prev_num_filters = lasagne.layers.get_output_shape(generator[prev_name])[1]
    generator[name] = lasagne.layers.Conv2DLayer(generator[prev_name], num_filters,
                                                filter_size, stride=1, pad='same', nonlinearity=lasagne.nonlinearities.tanh)
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(generator[name])
    details.append([name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                    str(output_dims)])
                    
    try:
        from tabulate import tabulate
        print tabulate(details)
    except ImportError:
        pass
    print("Number of parameters " + str(lasagne.layers.count_params(generator['gen_out'])))

    return generator, z_var


def build_discriminator(li, num_filters):

    input_var = T.tensor4('input_var')
    elu = lasagne.nonlinearities.elu

    discriminator = {}
    details = [['Layer Name', 'Dims in', 'shape of layer', 'Dims out']]
    filter_size = 3

    input_shape = (None, 3, li, li)
    name = 'input'
    discriminator[name] = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    output_dims = lasagne.layers.get_output_shape(discriminator[name])

    prev_name = name
    name = 'conv1'
    prev_num_filters = lasagne.layers.get_output_shape(discriminator[prev_name])[1]
    discriminator[name] = lasagne.layers.Conv2DLayer(discriminator[prev_name], num_filters,
                                                    filter_size, stride=1, pad='same', nonlinearity=elu)
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(discriminator[name])
    details.append([name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                    str(output_dims)])

    repeat_num = int(np.log2(np.array(li)) - 2)

    for n in range(2, 2 + repeat_num):
        filters = n*num_filters
        prev_name = name
        name = 'conv' + str(n)
        prev_num_filters = lasagne.layers.get_output_shape(discriminator[prev_name])[1]
        discriminator[name] = lasagne.layers.Conv2DLayer(discriminator[prev_name], filters,
                                                         filter_size, stride=1, pad='same', nonlinearity=elu)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(discriminator[name])
        details.append([name, str(prev_output_dims), str((filters, prev_num_filters, filter_size, filter_size)),
                        str(output_dims)])

        filters = n * num_filters
        prev_name = name
        name = 'conv' + str(n)
        prev_num_filters = lasagne.layers.get_output_shape(discriminator[prev_name])[1]
        discriminator[name] = lasagne.layers.Conv2DLayer(discriminator[prev_name], filters,
                                                         filter_size, stride=1, pad='same', nonlinearity=elu)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(discriminator[name])
        details.append([name, str(prev_output_dims), str((filters, prev_num_filters, filter_size, filter_size)),
                        str(output_dims)])

        prev_name = name
        name = 'conv' + str(n)
        prev_num_filters = lasagne.layers.get_output_shape(discriminator[prev_name])[1]
        discriminator[name] = lasagne.layers.Conv2DLayer(discriminator[prev_name], filters,
                                                         filter_size, stride=2, pad='same', nonlinearity=elu)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(discriminator[name])
        details.append([name, str(prev_output_dims), str((filters, prev_num_filters, filter_size, filter_size)),
                        str(output_dims)])

    # Fully connected layers
    prev_name = name
    name = 'fc_encode1'
    num_units = 8 * 8 * num_filters
    discriminator[name] = lasagne.layers.DenseLayer(discriminator[prev_name], num_units, nonlinearity=None)
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(discriminator[name])
    details.append([name, str(prev_output_dims), str((product(prev_output_dims[1:]), num_units)),
                    str(output_dims)])

    prev_name = name
    name = 'fc_encode2'
    num_units = 8 * 8 * num_filters
    discriminator[name] = lasagne.layers.DenseLayer(discriminator[prev_name], num_units, nonlinearity=None)
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(discriminator[name])
    details.append([name, str(prev_output_dims), str((product(prev_output_dims[1:]), num_units)),
                    str(output_dims)])

    prev_name = name
    name = 'reshape'
    discriminator[name] = lasagne.layers.ReshapeLayer(discriminator[prev_name], ([0], num_filters, 8, 8))
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(discriminator[name])
    details.append([name, str(prev_output_dims), str('reshape to 4D'),
                    str(output_dims)])

    scale_factor = 2
    for n in range(0, repeat_num):

        if n == 0:
            filters = num_filters
        else:
            filters = 2 * num_filters
        prev_name = name
        name = 'decode_conv' + str(n)
        prev_output_dims = output_dims
        prev_num_filters = prev_output_dims[1]
        discriminator[name] = lasagne.layers.Conv2DLayer(discriminator[prev_name], filters,
                                                         filter_size, stride=1, pad='same', nonlinearity=elu)
        output_dims = lasagne.layers.get_output_shape(discriminator[name])
        details.append([name, str(prev_output_dims), str((filters, prev_num_filters, filter_size, filter_size)),
                        str(output_dims)])

        filters = num_filters
        prev_name = name
        name = 'decode_conv' + str(n) + '_2'
        prev_num_filters = lasagne.layers.get_output_shape(discriminator[prev_name])[1]
        discriminator[name] = lasagne.layers.Conv2DLayer(discriminator[prev_name], filters,
                                                         filter_size, stride=1, pad='same', nonlinearity=elu)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(discriminator[name])
        details.append([name, str(prev_output_dims), str((filters, prev_num_filters, filter_size, filter_size)),
                        str(output_dims)])

        if n < repeat_num - 1:
            prev_name = name
            name = 'upscale_' + str(n)
            discriminator = nearest_neighbor(discriminator, name, prev_name, output_dims[2], scale_factor)
            output_dims = (None, num_filters, scale_factor*output_dims[2], scale_factor*output_dims[2])
            details.append([name, str(prev_output_dims), str('2x upscale'),
                            str(output_dims)])

    prev_name = name
    name = 'out'
    num_filters = 3
    discriminator[name] = lasagne.layers.Conv2DLayer(discriminator[prev_name], num_filters,
                                                     filter_size, stride=1, pad='same', nonlinearity=lasagne.nonlinearities.tanh)
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(discriminator[name])
    details.append([name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                    str(output_dims)])

    try:
        from tabulate import tabulate
        print(tabulate(details))
    except ImportError:
        pass
    print("Number of parameters " + str(lasagne.layers.count_params(discriminator['out'])))

    return discriminator, input_var
