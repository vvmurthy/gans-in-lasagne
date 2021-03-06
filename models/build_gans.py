import lasagne
import theano
import theano.tensor as T
import numpy as np


# helper function to decrease code length
def batch_conv(network, prev_name, name, num_filters, filter_size, stride, pad, W, nonlinearity):
    if W == False:
        network[name + '_conv'] = lasagne.layers.Conv2DLayer(network[prev_name], num_filters,
                    filter_size, stride=2, pad=pad, b=None, nonlinearity=None)
    else:
        network[name + '_conv'] = lasagne.layers.Conv2DLayer(network[prev_name], num_filters,
                    filter_size, stride=2, pad=pad, W=W, b=None, nonlinearity=None)
    
    # Batch Normalization
    network[name + '_bn'] = lasagne.layers.BatchNormLayer(network[name + '_conv'])
    
    # Then nonlinearity
    network[name] = lasagne.layers.NonlinearityLayer(network[name + '_bn'], nonlinearity=nonlinearity)
    
    return network
    
    
def make_train_fns(bz, li, nc, lab_ln, num_hidden):

    # defines variables
    print("Building model and compiling functions...")
    y_fake = T.tensor3('y_fake')
    y_3 = T.tensor3('y_3')
    y_var = T.tensor4('y_var')

    # Builds discriminator and generator
    # y_var is in format [batchsize, categories, 1] and is flattened out in build_generator
    discriminator, input_var = build_discriminator(y_var, li, nc, lab_ln)
    generator, z_var = build_generator(y_3, li, nc, lab_ln, num_hidden)
    
    # Important sections of code:
    #       "T.reshape..." - convert y_var to 4D tensor for use in discriminator
    #       "{discriminator['input']:..." - passes theano variables as inputs to input layers
    real_out = lasagne.layers.get_output(discriminator['out'], {discriminator['input']:input_var,
                                                                discriminator['input_y']:T.reshape(T.extra_ops.repeat(y_3,
                                                                (li/2)*(li/2), axis=2), (bz, lab_ln, (li/2), (li/2)))})
    real_out_yfake = lasagne.layers.get_output(discriminator['out'], {discriminator['input']:input_var,
                                                                      discriminator['input_y']:T.reshape(T.extra_ops.repeat(y_fake,
                                                                      (li/2)*(li/2), axis=2), (bz, lab_ln, (li/2), (li/2)))})
    ims = lasagne.layers.get_output(generator['gen_out'])
    fake_out = lasagne.layers.get_output(discriminator['out'], {discriminator['input']:ims,
                                                                discriminator['input_y']:T.reshape(T.extra_ops.repeat(y_3, (li/2)*(li/2), axis=2), (bz, lab_ln, (li/2), (li/2)))})

    # Create loss expressions
    two = theano.compile.shared(2, allow_downcast=True)
    generator_loss = lasagne.objectives.binary_crossentropy(fake_out, 1).mean()
    discriminator_loss = lasagne.objectives.binary_crossentropy(real_out, 1).mean() + (lasagne.objectives.binary_crossentropy(real_out_yfake, 0).mean() +
                          lasagne.objectives.binary_crossentropy(fake_out, 0).mean()) / two

    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator['gen_out'], trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator['out'], trainable=True)
    lr = T.fscalar('lr')
    gen_updates = lasagne.updates.adam(
        generator_loss, generator_params, learning_rate=lr, beta1=0.5)
    dis_updates = lasagne.updates.adam(discriminator_loss, discriminator_params, learning_rate=lr, beta1=0.5)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    gen_train_fn = theano.function([z_var, y_3, lr], [fake_out, ims, (fake_out < .5).mean()], updates=gen_updates)
    dis_train_fn = theano.function([input_var, z_var, y_3, y_fake, lr], [real_out, real_out_yfake, (real_out < .5).mean(), (fake_out > .5).mean()], updates=dis_updates)

    # Compile another function generating some data
    gen_fn = theano.function([z_var, y_3], lasagne.layers.get_output(generator['gen_out'], deterministic=True))
    print("...Done")    
    return generator, discriminator, gen_train_fn, gen_fn, dis_train_fn


def build_generator(y_var, li, nc, lab_ln, num_hidden):

    z_var = T.fmatrix('z_var')
    generator = {}
    details = [['Layer Name', 'Dims in', 'shape of layer', 'Dims out']]

    # input noise
    name = 'gen_z_var'
    input_shape = (None, num_hidden)
    generator[name] = lasagne.layers.InputLayer(shape=input_shape, input_var=z_var)

    # Input: y vector
    name = 'gen_y_var'
    input_shape = (None, lab_ln, 1)
    generator[name] = lasagne.layers.InputLayer(shape=input_shape, input_var=y_var)
    
    # reshape y_var to 2D
    prev_name = name
    name = 'gen_reshape_y'
    generator[name] = lasagne.layers.ReshapeLayer(generator[prev_name], ([0], -1))

    # Concatenate y variable
    name = 'gen_concat'
    generator[name] = lasagne.layers.ConcatLayer([generator['gen_z_var'], generator['gen_reshape_y']])
    output_dims = lasagne.layers.get_output_shape(generator[name])
    
    # Reshape to batchsize x length x 1 x 1
    prev_name = name
    name = 'gen_reshape'
    generator[name] = lasagne.layers.ReshapeLayer(generator[prev_name], ([0], num_hidden + lab_ln, 1, 1))
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(generator[name])
    details.append([name, str(prev_output_dims), str('reshape to 4D'),
                    str(output_dims)])

    # First convolution (uses valid, not full padding)
    prev_name = name
    name = 'gen_conv0_0'
    num_filters = 2 ** int(np.log2(np.array(li)) + 3)    
    filter_size = 4
    prev_num_filters = lasagne.layers.get_output_shape(generator[prev_name])[1]
    generator[name] = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(generator[prev_name], num_filters,
                                                                             filter_size, stride=1, crop='valid'))
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(generator[name])
    details.append([name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                    str(output_dims)])

    # Constructs batch norm convolutions
    # each convolution is preceded by padding
    # Pad with zeros on all sides (from torch implementation)
    # We repeat 3 times for 64 pixels
    repeat_num = int(np.log2(np.array(li)) - 3)
    for n in range(0, repeat_num):
        prev_name = name
        name = 'gen_pad' + str(n)
        pad_amt = 1
        generator[name] = lasagne.layers.PadLayer(generator[prev_name], pad_amt)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(generator[name])
        details.append([name, str(prev_output_dims), str(pad_amt),
                        str(output_dims)])

        prev_name = name
        name = 'gen_conv' + str(n)
        num_filters = num_filters / 2
        filter_size = 4
        prev_num_filters = lasagne.layers.get_output_shape(generator[prev_name])[1]
        generator[name] = lasagne.layers.batch_norm(lasagne.layers.Deconv2DLayer(generator[prev_name], num_filters,
                                                                                 filter_size, stride=2, crop='full'))
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(generator[name])
        details.append([name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                        str(output_dims)])
                    
    prev_name = name
    name = 'gen_pad_out'
    pad_amt = 1
    generator[name] = lasagne.layers.PadLayer(generator[prev_name], pad_amt)
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(generator[name])
    details.append([name, str(prev_output_dims), str(pad_amt),
                    str(output_dims)])

    prev_name = name
    name = 'gen_out'
    num_filters = nc
    filter_size = 4
    prev_num_filters = lasagne.layers.get_output_shape(generator[prev_name])[1]
    generator[name] = lasagne.layers.Deconv2DLayer(generator[prev_name], num_filters,
                                                filter_size, stride=2, crop='full', nonlinearity=lasagne.nonlinearities.tanh)
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(generator[name])
    details.append([name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                    str(output_dims)])
                    
    try:
        from tabulate import tabulate
        print tabulate(details)
    except ImportError:
        pass

    return generator, z_var


def build_discriminator(y_var, li, nc, lab_ln):

    input_var = T.tensor4('input_var')
    lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

    discriminator = {}
    details = [['Layer Name', 'Dims in', 'shape of layer', 'Dims out']]

    input_shape = (None, lab_ln, li/2, li/2)
    name = 'input_y'
    discriminator[name] = lasagne.layers.InputLayer(shape=input_shape, input_var=y_var)
    output_dims = lasagne.layers.get_output_shape(discriminator[name])

    input_shape = (None, nc, li, li)
    name = 'input'
    discriminator[name] = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    output_dims = lasagne.layers.get_output_shape(discriminator[name])
    
    prev_name = name
    name = 'pad1'
    pad_amt = 1
    discriminator[name] = lasagne.layers.PadLayer(discriminator[prev_name], pad_amt)
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(discriminator[name])
    details.append([name, str(prev_output_dims), str(pad_amt),
                    str(output_dims)])

    prev_name = name
    name = 'conv1'
    num_filters = li
    filter_size = 4
    prev_num_filters = lasagne.layers.get_output_shape(discriminator[prev_name])[1]
    discriminator[name] = lasagne.layers.Conv2DLayer(discriminator[prev_name], num_filters,
                                                    filter_size, stride=2, pad='valid', nonlinearity=lrelu)
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(discriminator[name])
    details.append([name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                    str(output_dims)])

    prev_name = name
    name = 'concat'
    discriminator[name] = lasagne.layers.ConcatLayer([discriminator[prev_name], discriminator['input_y']])
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(discriminator[name])
    details.append([name, str(prev_output_dims), str('concat'),
                    str(output_dims)])

    repeat_num = int(np.log2(np.array(li)) - 3)

    for n in range(2, repeat_num + 2):

        prev_name = name
        name = 'pad' + str(n)
        pad_amt = 1
        discriminator[name] = lasagne.layers.PadLayer(discriminator[prev_name], pad_amt)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(discriminator[name])
        details.append([name, str(prev_output_dims), str(pad_amt),
                        str(output_dims)])

        prev_name = name
        name = 'conv' + str(n)
        num_filters = num_filters * 2
        filter_size = 4
        prev_num_filters = lasagne.layers.get_output_shape(discriminator[prev_name])[1]
        discriminator = batch_conv(discriminator, prev_name, name, num_filters, filter_size,
                               2, 'valid', False, lrelu)
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(discriminator[name])
        details.append([name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                        str(output_dims)])

    prev_name = name
    name = 'conv5'
    filter_size = 4
    num_filters = 1
    prev_num_filters = lasagne.layers.get_output_shape(discriminator[prev_name])[1]
    discriminator[name] = lasagne.layers.Conv2DLayer(discriminator[prev_name], num_filters,
                                                                               filter_size, stride=1, pad='valid',
                                                                               nonlinearity=lasagne.nonlinearities.sigmoid)
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(discriminator[name])
    details.append([name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                    str(output_dims)])
    
    prev_name = name
    name = 'out'
    discriminator[name] = lasagne.layers.ReshapeLayer(discriminator[prev_name], ([0], -1))
    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(discriminator[name])
    details.append([name, str(prev_output_dims), str('reshape'),
                    str(output_dims)])

    try:
        from tabulate import tabulate
        print(tabulate(details))
    except ImportError:
        pass

    return discriminator, input_var