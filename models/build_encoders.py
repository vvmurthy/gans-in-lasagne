import lasagne
import theano
import theano.tensor as T
import numpy as np

from utils.gen_utils import product


def build_encoder_y(li, nc, lab_ln, lr):
    y_var = T.fmatrix('y_var')
    input_var = T.tensor4('inputs')
    encoder = {}
    details = [['Layer Name', 'Dims in', 'shape of layer', 'Dims out']]

    input_shape = (None, nc, li, li)
    name = 'input'
    encoder[name] = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    output_dims = input_shape
    filter_size = 5
    num_filters = li / 4
    
    repeat_num = int(np.log2(np.array(li)) - 3) + 1
    
    for n in range(0, repeat_num):
        num_filters = num_filters * 2
        prev_name = name
        name = 'conv' + str(n)
        prev_num_filters = lasagne.layers.get_output_shape(encoder[prev_name])[1]
        encoder[name] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(encoder[prev_name], num_filters,
                                                                           filter_size, stride=2, pad='same',
                                                                           nonlinearity=lasagne.nonlinearities.rectify))
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(encoder[name])
        details.append([name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                    str(output_dims)])

    prev_name = name
    name = 'fc'
    num_units = int(li * li / 8)

    encoder[name] = lasagne.layers.DenseLayer(encoder[prev_name],
                                              num_units=num_units,
                                              nonlinearity=lasagne.nonlinearities.rectify)

    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(encoder[name])

    details.append([name, str(prev_output_dims), str((product(prev_output_dims[1:]), num_units)),
                    str(output_dims)])

    prev_name = name
    name = 'out'
    num_units = lab_ln

    encoder[name] = lasagne.layers.DenseLayer(encoder[prev_name],
                                              num_units=num_units,
                                              nonlinearity=lasagne.nonlinearities.tanh)

    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(encoder[name])

    details.append([name, str(prev_output_dims), str((product(prev_output_dims[1:]), num_units)),
                    str(output_dims)])

    train_out = lasagne.layers.get_output(encoder['out'])
    val_out = lasagne.layers.get_output(encoder['out'], deterministic=True)

    loss = lasagne.objectives.squared_error(train_out, y_var).mean()
    params = lasagne.layers.get_all_params(encoder['out'], trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=lr, beta1=0.5)
    train_fn = theano.function([input_var, y_var], [loss], updates=updates)
    val_fn = theano.function([input_var], [val_out])

    try:
        from tabulate import tabulate
        print(tabulate(details))
    except ImportError:
        pass
    return encoder, train_fn, val_fn


def build_encoder_z(li, nc, lr):
    z_var = T.fmatrix('z_var')
    input_var = T.tensor4('inputs')
    encoder = {}
    details = [['Layer Name', 'Dims in', 'shape of layer', 'Dims out']]

    input_shape = (None, nc, li, li)
    name = 'input'
    encoder[name] = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    output_dims = input_shape

    filter_size = 5
    num_filters = li / 4
    
    repeat_num = int(np.log2(np.array(li)) - 3) + 1
    
    for n in range(0, repeat_num):
        num_filters = num_filters * 2
        prev_name = name
        name = 'conv' + str(n)
        prev_num_filters = lasagne.layers.get_output_shape(encoder[prev_name])[1]
        encoder[name] = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(encoder[prev_name], num_filters,
                                                                           filter_size, stride=2, pad='same',
                                                                           nonlinearity=lasagne.nonlinearities.rectify))
        prev_output_dims = output_dims
        output_dims = lasagne.layers.get_output_shape(encoder[name])
        details.append([name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                    str(output_dims)])

    prev_name = name
    name = 'fc'
    num_units = int(li * li)

    encoder[name] = lasagne.layers.DenseLayer(encoder[prev_name],
                                                    num_units=num_units,
                                                    nonlinearity=lasagne.nonlinearities.rectify)

    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(encoder[name])
    details.append([name, str(prev_output_dims), str((product(prev_output_dims[1:]), num_units)),
                    str(output_dims)])

    prev_name = name
    name = 'out'
    num_units = 100
    
    # We restrict output to tanh domain (same as input noise)
    encoder[name] = lasagne.layers.DenseLayer(encoder[prev_name],
                                              num_units=num_units,
                                              nonlinearity=lasagne.nonlinearities.tanh)

    prev_output_dims = output_dims
    output_dims = lasagne.layers.get_output_shape(encoder[name])
    details.append([name, str(prev_output_dims), str((product(prev_output_dims[1:]), num_units)),
                    str(output_dims)])

    train_out = lasagne.layers.get_output(encoder['out'])
    val_out = lasagne.layers.get_output(encoder['out'], deterministic=True)

    loss = lasagne.objectives.squared_error(train_out, z_var).mean()
    params = lasagne.layers.get_all_params(encoder['out'], trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=lr, beta1=0.5)
    train_fn = theano.function([input_var, z_var], [loss], updates=updates)
    val_fn = theano.function([input_var], [val_out])

    try:
        from tabulate import tabulate
        print(tabulate(details))
    except ImportError:
        pass
    return encoder, train_fn, val_fn
