import os
import time

import lasagne
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from src.utils.gen_utils import *
from src.utils.graphs_conditional import *
from src.utils.iters import *
from src.utils.layers import *
from src.utils.z_vars import *

from src.utils.graphs_unconditional import *


class IcGAN:
    """
    models.IcGAN(dataset, folder_name='icgan', **kwargs)
    IcGAN initialization and training class. Trains IcGAN model from [1].

    Our implementation is based off of [2] and [3].

    Our implementation of ICGAN is trained in two steps.
        1. Train the Generator + Discriminator
        2. Train Encoder Z + Y, using minibatches of generated images from the trained generator

    This is in contrast with the author's implementation that generates a single set of images to train on.

    On a NVIDIA GTX 1060 6GB GPU, the GAN models took about 15 minutes per epoch to train.
    The encoders took about 4-5 minutes per epoch to train.

    Parameters
    ----------
    dataset : :class:`Dataset` Object
        Which dataset class to use. Provides dataset-specific
        hyperparameters + functions.
    folder_name : string
        Where to store generated models, images, stats on training.
         Stores to ``os.getcwd() + '/' + 'folder_name'``
    Attributes
    ----------
    start_lr : float
        The initial learning rate to use. Default is 0.0002.
    num_epochs : int
         The number of epochs (full passes over dataset) to perform.
         Default is 25.
    bz : int
         The batchsize to use. Batches are moved from memory to GPU.
         Default size is 64.
    num_hidden : int
         The number of units to use in hidden representation of image.
         Default is 100.
    z_var : Function
        Method to generated noise vector. Default is to sample from
        uniform distribution.
    gamma : float
        Gamma from Berthelot et al. Default is 0.7, though the authors
        sample from 0.5 up.
    num_examples : int
        The number of example images to generate. Each epoch results in num_examples ** 2
        generated images being turned into a figure and saved.
    seed : int
        Seed to use for random number (z_vector) generation during test time. Default is 126.
    n_inter : int
        Number of steps to use when generating interpolation test figure.
    References
    ----------
    .. [1] Guim Perarnau, Joost van de Weijer, Bogdan Raducanu,
           Jose M. Alvarez(2016):
           Invertible Conditional GANs for Image Editing. arXiv.
           https://arxiv.org/abs/1611.06355
    .. [2] https://github.com/Guim3/IcGAN
    .. [3] https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56

    """
    def __init__(self, folder_name, dataset, **kwargs):

        # set hyperparameters
        self.start_lr = kwargs.get('lr', 0.0002)
        self.lr = self.start_lr
        self.num_epochs = kwargs.get('self.num_epochs', 25)
        self.bz = kwargs.get('bz', 64)
        self.num_hidden = kwargs.get('num_hidden', 100)
        self.folder_name = folder_name
        self.z_var = kwargs.get('z_var', z_var_uniform)
        self.num_examples = kwargs.get('num_examples', 100)
        self.seed = kwargs.get('seed', 115)

        # set variables from dataset
        self.dataset = dataset
        self.X_files_train = dataset.X_files_train
        self.X_files_val = dataset.X_files_val
        self.X_files_test = dataset.X_files_test
        self.y_train = dataset.y_train
        self.y_val = dataset.y_val
        self.y_test = dataset.y_test
        self.li = dataset.li
        self.lab_ln = dataset.lab_ln
        self.labels = dataset.labels
        self.nc = dataset.nc
        self.images_in_mem = dataset.images_in_mem
        self.randomize_y = dataset.randomize_y
        self.modify_y = dataset.modify_y

        # Perform variable checks
        if 2 ** int(np.log2(self.li)) != self.li:
            raise ValueError(
                "Image length for IcGAN must be power of 2"
            )

        self.base = os.getcwd() + '/' + folder_name + '/'
        if not os.path.isdir(self.base):
            os.mkdir(self.base)
            os.mkdir(self.base + 'models/')
            os.mkdir(self.base + 'images/')
            os.mkdir(self.base + 'stats/')

    def build_encoder_y(self, li, nc, lab_ln, lr):
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
            details.append(
                [name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
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

    def build_encoder_z(self, li, nc, num_hidden, lr):
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
            details.append(
                [name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
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
        num_units = num_hidden

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

    def build_generator(self, y_var, li, nc, lab_ln, num_hidden):

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
                                                                                 filter_size, stride=1,
                                                                                 crop='valid'))
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
            generator[name] = lasagne.layers.batch_norm(
                lasagne.layers.Deconv2DLayer(generator[prev_name], num_filters,
                                             filter_size, stride=2, crop='full'))
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(generator[name])
            details.append(
                [name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
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
                                                       filter_size, stride=2, crop='full',
                                                       nonlinearity=lasagne.nonlinearities.tanh)
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

    def build_discriminator(self, y_var, li, nc, lab_ln):

        input_var = T.tensor4('input_var')
        lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

        discriminator = {}
        details = [['Layer Name', 'Dims in', 'shape of layer', 'Dims out']]

        input_shape = (None, lab_ln, li / 2, li / 2)
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
        name = 'conv_init'
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
        for n in range(0, repeat_num):
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
            details.append(
                [name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                 str(output_dims)])

        prev_name = name
        name = 'conv_final'
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

    def build_train_fns(self):

        print("Building model and compiling functions...")
        y_fake = T.tensor3('y_fake')
        y_3 = T.tensor3('y_3')
        y_var = T.tensor4('y_var')

        # Builds discriminator and generator
        # y_var is in format [batchsize, categories, 1] and is flattened out in build_generator
        discriminator, input_var = self.build_discriminator(y_var, self.li, self.nc, self.lab_ln)
        generator, z_var = self.build_generator(y_3, self.li, self.nc, self.lab_ln, self.num_hidden)

        real_out = lasagne.layers.get_output(discriminator['out'],
                                             {discriminator['input']: input_var,
                                              discriminator['input_y']: T.reshape(
                                                                        T.extra_ops.repeat(y_3,
                                                                                           (self.li / 2) * (self.li / 2), axis=2),
                                                                        (self.bz, self.lab_ln, (self.li / 2), (self.li / 2)))})
        real_out_yfake = lasagne.layers.get_output(discriminator['out'], {discriminator['input']: input_var,
                                                                          discriminator['input_y']: T.reshape(
                                                                              T.extra_ops.repeat(y_fake,
                                                                                                 (self.li / 2) * (self.li / 2),
                                                                                                 axis=2),
                                                                              (self.bz, self.lab_ln, (self.li / 2), (self.li / 2)))})
        ims = lasagne.layers.get_output(generator['gen_out'])
        fake_out = lasagne.layers.get_output(discriminator['out'], {discriminator['input']: ims,
                                                                    discriminator['input_y']: T.reshape(
                                                                        T.extra_ops.repeat(y_3, (self.li / 2) * (self.li / 2),
                                                                                           axis=2),
                                                                        (self.bz, self.lab_ln, (self.li / 2), (self.li / 2)))})

        # Create loss expressions
        two = theano.compile.shared(2, allow_downcast=True)
        generator_loss = lasagne.objectives.binary_crossentropy(fake_out, 1).mean()
        discriminator_loss = lasagne.objectives.binary_crossentropy(real_out, 1).mean() + (
                                                                                          lasagne.objectives.binary_crossentropy(
                                                                                              real_out_yfake,
                                                                                              0).mean() +
                                                                                          lasagne.objectives.binary_crossentropy(
                                                                                              fake_out, 0).mean()) / two

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
        dis_train_fn = theano.function([input_var, z_var, y_3, y_fake, lr],
                                       [real_out, real_out_yfake, (real_out < .5).mean(), (fake_out > .5).mean()],
                                       updates=dis_updates)

        # Compile another function generating some data
        gen_fn = theano.function([z_var, y_3], lasagne.layers.get_output(generator['gen_out'], deterministic=True))

        # Build encoders
        encoder_z, train_fn_z, val_fn_z = build_encoder_z(self.li, self.nc, self.num_hidden, self.lr)
        encoder_y, train_fn_y, val_fn_y = build_encoder_y(self.li, self.nc, self.lab_ln, self.lr)

        print("...Done")
        return generator, discriminator, encoder_z, encoder_y, \
            gen_train_fn, gen_fn, dis_train_fn, \
            train_fn_z, val_fn_z, \
            train_fn_y, val_fn_y

    def train(self):
        # Make training functions
        print("Making Training Functions...")
        generator, discriminator, encoder_z, encoder_y, \
        gen_train_fn, gen_fn, dis_train_fn, \
        encoder_z_train, encoder_z_test, \
        encoder_y_train, encoder_y_test = self.build_train_fns()

        # Load in params if training incomplete
        try:

            # Load training statistics
            start_epoch = np.load(self.base + 'stats/epoch.npy')[0]
            gen_train_err = np.load(self.base + 'stats/gen_train_err.npy')
            dis_train_err_real = np.load(self.base + 'stats/dis_train_err_real.npy')
            dis_train_err_fake = np.load(self.base + 'stats/dis_train_err_fake.npy')

            # Load models
            with np.load(self.base + 'models/generator_epoch' + str(start_epoch) + '.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(generator['gen_out'], param_values)

            with np.load(self.base + 'models/discriminator_epoch' + str(start_epoch) + '.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(discriminator['out'], param_values)

            start_epoch += 1
            print("...Loaded previous models")
        except IOError:
            start_epoch = 0
            gen_train_err = np.zeros((self.num_epochs)).astype(np.float32)
            dis_train_err_real = np.zeros((self.num_epochs)).astype(np.float32)
            dis_train_err_fake = np.zeros((self.num_epochs)).astype(np.float32)
            print("...Loaded models")

        print("Starting cGAN Training...")
        for epoch in range(start_epoch, self.num_epochs):
            start_time = time.time()
            dis_train_real = 0
            dis_train_fake = 0
            gen_ = 0

            # Train cGAN
            num_batches = 0
            # X_files_train = X_files_train[0: (X_files_train.shape[0] / bz) * bz]
            # y_train = y_train[0: (y_train.shape[0] / bz) * bz, :]

            # Load specified amount of images into memory at once
            for X_files_mem, y_train_mem in iterate_membatches(self.X_files_train,
                                                               self.y_train,
                                                               self.images_in_mem,
                                                               self.dataset.load_files,
                                                               shuffle=True):

                for batch in iterate_minibatches_conditional(X_files_mem, y_train_mem, self.bz, shuffle=True):
                    inputs, targets = batch

                    # expands dims for training (generator + discriminator expect 3D input)
                    targets = np.expand_dims(targets, 2)

                    # Create noise vector
                    noise = self.z_var(self.bz, self.num_hidden)
                    y_fake = self.randomize_y(targets)

                    # Train the generator
                    fake_out, ims, gen_train_err_epoch = gen_train_fn(noise, targets, self.lr)
                    gen_ += gen_train_err_epoch

                    # Train the discriminator
                    #   + real_out - predictions on real images + matching y vectors
                    #   + real_out_yfake - predictions on real_images + fake y vectors
                    real_out, real_out_yfake, dis_train_err_real_epoch, dis_train_err_fake_epoch = dis_train_fn(inputs,
                                                                                                                noise,
                                                                                                                targets,
                                                                                                                y_fake,
                                                                                                                self.lr)
                    dis_train_real += dis_train_err_real_epoch
                    dis_train_fake += dis_train_err_fake_epoch

                    num_batches += 1

            # Display training stats
            gen_train_err[epoch] = gen_
            dis_train_err_real[epoch] = dis_train_real
            dis_train_err_fake[epoch] = dis_train_fake
            print("Epoch {} of {} took {:.3f} minutes".format(epoch + 1, self.num_epochs,
                                                              (time.time() - start_time) / np.float32(60)))
            print("  Generator Error:\t\t{}".format(gen_train_err[epoch] / num_batches))
            print("  Discriminator Error on real ims:\t\t{}".format(dis_train_err_real[epoch] / num_batches))
            print("  Discriminator Error on fake ims:\t\t{}".format(dis_train_err_fake[epoch] / num_batches))

            # Save stats + models
            np.save(self.base + 'stats/epoch.npy', np.array([epoch]))
            np.save(self.base + 'stats/gen_train_err.npy', gen_train_err)
            np.save(self.base + 'stats/dis_train_err_real.npy', dis_train_err_real)
            np.save(self.base + 'stats/dis_train_err_fake.npy', dis_train_err_fake)
            np.savez(self.base + 'models/generator_epoch' + str(epoch) + '.npz',
                     *lasagne.layers.get_all_param_values(generator['gen_out']))
            np.savez(self.base + 'models/discriminator_epoch' + str(epoch) + '.npz',
                     *lasagne.layers.get_all_param_values(discriminator['out']))

            # Decay the lr
            if epoch >= self.num_epochs // 2:
                progress = float(epoch) / self.num_epochs
                self.lr = self.start_lr * 2 * (1 - progress)

            # Do a pass over first 10 sets of 64 y vectors from vaself.lidation set every epoch, show example images
            # note many example images are repeated between categories
            sets = 10
            val_ims = np.zeros((self.bz * sets, self.nc, self.li, self.li))
            for st in range(0, sets):
                noise = self.z_var(self.bz, self.num_hidden)
                targets = np.expand_dims(self.y_val[self.bz * st: self.bz * st + self.bz], 2).astype(np.float32)
                val_ims[self.bz * st: self.bz * st + self.bz] = gen_fn(noise, targets)

            show_examples(val_ims, self.y_val[0:self.bz * sets], self.labels, self.li, self.nc, epoch,
                          self.base + 'images/epoch' + str(epoch) + '.png')

        # Make graph with training statistics
        show_training_stats_graph(gen_train_err, dis_train_err_real, dis_train_err_fake, self.num_epochs,
                                  self.base + 'stats/stats_graph.png')

        # Save final models
        np.savez(self.base + 'models/generator_final.npz', *lasagne.layers.get_all_param_values(generator['gen_out']))
        np.savez(self.base + 'models/discriminator_final.npz', *lasagne.layers.get_all_param_values(discriminator['out']))

        print("...cGAN Training Complete")

        # Load in params if partial training
        try:

            # Load training statistics
            start_epoch = np.load(self.base + 'stats/encoder_epoch.npy')[0]
            encoder_z_loss = np.load(self.base + 'stats/encoder_z_loss.npy')
            encoder_y_loss = np.load(self.base + 'stats/encoder_y_loss.npy')

            # Load models
            with np.load(self.base + 'encoder_z_epoch' + str(start_epoch) + '.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(encoder_z, param_values)

            with np.load(self.base + 'encoder_y_epoch' + str(start_epoch) + '.npz') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(encoder_y, param_values)

            start_epoch += 1
            print("...Loaded partially trained models")
        except IOError:
            start_epoch = 0
            encoder_y_loss = np.zeros((self.num_epochs)).astype(np.float32)
            encoder_z_loss = np.zeros((self.num_epochs)).astype(np.float32)
            print("...Built models")

        # Make some generated samples for training encoder
        # Train encoder as well
        for epoch in range(start_epoch, self.num_epochs):
            start_time = time.time()
            num_batches = 0
            for batch in iterate_minibatches_conditional(self.X_files_train,
                                                         self.y_train,
                                                         self.bz,
                                                         shuffle=True):
                _, targets = batch
                targets = targets.astype(np.float32)
                gen_targets = np.expand_dims(targets, 2).astype(np.float32)
                noise = self.z_var(self.bz, self.num_hidden)
                gen_images = gen_fn(noise, gen_targets)
                encoder_z_loss[epoch] += encoder_z_train(gen_images, noise)
                encoder_y_loss[epoch] += encoder_y_train(gen_images, targets)
                num_batches += 1

            # Print training stats
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.num_epochs, time.time() - start_time))
            print("  Encoder Z loss:\t\t{:.3f}".format(encoder_z_loss[epoch] / num_batches))
            print("  Encoder y loss:\t\t{:.3f}".format(encoder_y_loss[epoch] / num_batches))

            # Show sample reconstructions from y val
            num_examples = 10
            ims = self.dataset.load_files(self.X_files_val[0: num_examples], num_examples, self.li)
            z = encoder_z_test(ims)[0]
            y = np.expand_dims(encoder_y_test(ims)[0], 2)
            reconstructions = gen_fn(z, y)
            show_reconstructions(ims, reconstructions, self.li, self.nc, epoch,
                                 self.base + 'images/reconstructions_epoch' + str(epoch) + '.png')

            # Save training stats and intermediate models
            np.save(self.base + 'stats/encoder_epoch.npy', np.array([epoch]))
            np.save(self.base + 'stats/encoder_z_loss.npy', encoder_z_loss)
            np.save(self.base + 'stats/encoder_y_loss.npy', encoder_y_loss)
            np.savez(self.base + 'models/encoder_z_epoch' + str(epoch) + '.npz',
                     *lasagne.layers.get_all_param_values(encoder_z['out']))
            np.savez(self.base + 'models/encoder_y_epoch' + str(epoch) + '.npz',
                     *lasagne.layers.get_all_param_values(encoder_y['out']))

        # Save encoder (final)
        np.savez(self.base + 'models/encoder_z_final.npz', *lasagne.layers.get_all_param_values(encoder_z['out']))
        np.savez(self.base + 'models/encoder_y_final.npz', *lasagne.layers.get_all_param_values(encoder_y['out']))

        # Make training stats graph
        show_encoder_stats_graph(encoder_z_loss, encoder_y_loss, self.num_epochs, self.base + 'stats/encoder_stats.png')

    def test_icgan(self):

        # seed indices generation
        np.random.seed(self.seed)

        # Check that models do exist
        assert (os.path.isfile(self.base + '/models/generator_final.npz')), \
            ("Generator in " + self.base + " Does not exist")
        assert (os.path.isfile(self.base + '/models/discriminator_final.npz')), \
            ("Discriminator in " + self.base + " Does not exist")
        assert (os.path.isfile(self.base + '/models/encoder_z_final.npz')), \
            ("Encoder Z in " + self.base + " Does not exist")
        assert (os.path.isfile(self.base + '/models/encoder_y_final.npz')), \
            ("Encoder y in " + self.base + " Does not exist")

        # Build GAN + Encoder
        generator, discriminator, encoder_z, encoder_y, \
        gen_train_fn, gen_fn, dis_train_fn, \
        encoder_z_train, encoder_z_test, \
        encoder_y_train, encoder_y_test = self.build_train_fns()

        # Set params
        with np.load(self.base + '/models/generator_final.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(generator['gen_out'], param_values)

        with np.load(self.base + '/models/discriminator_final.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(discriminator['out'], param_values)

        with np.load(self.base + '/models/encoder_z_final.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(encoder_z['out'], param_values)

        with np.load(self.base + '/models/encoder_y_final.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(encoder_y['out'], param_values)

        num_people = 10  # use 4 people for reconstruction

        # Reconstruct + change attributes
        # Torch implementation gets running train batch norms std and mean, this uses fixed vals
        all_reconstructions = np.zeros((self.li * num_people,
                                        self.li * (self.lab_ln + 2), self.nc)).astype(np.float32)
        indices = np.random.randint(0, self.X_files_test.shape[0], num_people)
        for index in range(0, num_people):
            image = self.dataset.load_files(self.X_files_test[indices[index]], 1, self.li)
            y_pred = np.squeeze(encoder_y_test(image)[0])
            z = np.squeeze(encoder_z_test(image)[0])

            # Duplicate Z
            z_permutations = np.zeros((self.lab_ln + 1, 100)).astype(np.float32)
            for n in range(0, z_permutations.shape[0]):
                z_permutations[n, :] = z

            # Create y matrix
            y_permutations = np.expand_dims(self.modify_y(y_pred, True), axis=2)

            # Generate images
            generated_ims = gen_fn(z_permutations, y_permutations)

            # Map reconstructions to main image
            all_reconstructions[self.li * index: self.li * index + self.li,
            0: self.li, :] = deprocess_image(image[0, :, :, :], self.li, self.nc)

            for n in range(0, generated_ims.shape[0]):
                all_reconstructions[self.li * index: self.li * index + self.li,
                self.li * (n + 1): self.li * (n + 1) + self.li,
                :] = deprocess_image(generated_ims[n, :, :, :], self.li, self.nc)

        # Plot the reconstruction
        fig, ax = plt.subplots()

        ax.set_yticks([])
        ax.set_xticks(np.arange(0, self.li * len(self.labels) + 2 * self.li, self.li) + (self.li / 2), minor=False)

        ax.set_title("Sample Generated Images")

        ax.set_xticklabels(['Original', 'Reconstruction'] + self.labels, rotation='vertical', minor=False)
        ax.set_yticklabels([])

        if self.nc == 1:
            plt.imshow(np.squeeze(all_reconstructions), cmap='gray')
        else:
            plt.imshow(all_reconstructions)

        fig.savefig(self.base + '/images/reconstructions.png', bbox_inches='tight')
        plt.close(fig)

        # Swap
        indices = np.random.randint(0, self.X_files_test.shape[0], 2)
        image_1 = self.dataset.load_files(self.X_files_test[indices[0]], 1, self.li)
        image_2 = self.dataset.load_files(self.X_files_test[indices[1]], 1, self.li)
        y_1 = np.squeeze(encoder_y_test(image_1)[0])
        z_1 = np.squeeze(encoder_z_test(image_1)[0])
        y_2 = np.squeeze(encoder_y_test(image_2)[0])
        z_2 = np.squeeze(encoder_z_test(image_2)[0])

        swap_image = np.zeros((self.li * 2, self.li * 3, self.nc)).astype(np.float32)
        swap_image[0:self.li, 0:self.li, :] = deprocess_image(image_1[0, :, :, :], self.li, self.nc)
        swap_image[self.li:self.li + self.li, 0:self.li, :] = deprocess_image(image_2[0, :, :, :], self.li, self.nc)

        # Swaps for first image
        z_matrix = np.zeros((2, self.num_hidden)).astype(np.float32)
        y_matrix = np.zeros((2, self.lab_ln)).astype(np.float32)
        for n in range(0, 2):
            z_matrix[n, :] = z_1

        y_matrix[0, :] = y_1
        y_matrix[1, :] = y_2

        ims = gen_fn(z_matrix, np.expand_dims(y_matrix, axis=2))
        for n in range(1, 3):
            swap_image[0:self.li, n * self.li:n * self.li + self.li, :] = \
                deprocess_image(ims[n - 1, :, :, :], self.li, self.nc)

        # Swaps for second image
        z_matrix = np.zeros((2, self.num_hidden)).astype(np.float32)
        y_matrix = np.zeros((2, self.lab_ln)).astype(np.float32)
        for n in range(0, 2):
            z_matrix[n, :] = z_2

        y_matrix[0, :] = y_2
        y_matrix[1, :] = y_1

        ims = gen_fn(z_matrix, np.expand_dims(y_matrix, axis=2))
        for n in range(1, 3):
            swap_image[self.li : self.li + self.li, n * self.li:n * self.li + self.li, :] = \
                deprocess_image(ims[n - 1, :, :, :], self.li, self.nc)

        # Plot the swapped images
        x_lab = ['Original', 'Reconstruction', 'Swapped y']
        fig, ax = plt.subplots(figsize=(3, 2))

        ax.set_yticks([])
        ax.set_xticks(np.arange(0, self.li * 3, self.li) + (self.li / 2), minor=False)
        ax.invert_yaxis()

        ax.set_title("Swapped Images")

        ax.set_xticklabels(x_lab, rotation='vertical', minor=False)
        ax.set_yticklabels([])

        if self.nc == 1:
            plt.imshow(np.squeeze(swap_image), cmap='gray')
        else:
            plt.imshow(swap_image)

        fig.savefig(self.base + '/images/swapped.png', bbox_inches='tight')
        plt.close(fig)

        # Interpolation
        n_inter = 10
        indices = np.random.randint(0, self.X_files_test.shape[0], num_people)
        interpolations = np.zeros((self.li * num_people / 2, self.li * n_inter + self.li, self.nc)).astype(np.float32)
        for n in range(0, num_people, 2):

            image_1 = self.dataset.load_files(self.X_files_test[indices[n]], 1, self.li)
            image_2 = self.dataset.load_files(self.X_files_test[indices[n + 1]], 1, self.li)
            y_1 = np.squeeze(encoder_y_test(image_1)[0])
            z_1 = np.squeeze(encoder_z_test(image_1)[0])
            y_2 = np.squeeze(encoder_y_test(image_2)[0])
            z_2 = np.squeeze(encoder_z_test(image_2)[0])

            # Interpolate y and z
            y_inter = np.expand_dims(interpolate_vector(y_1, y_2, n_inter), axis=2)
            z_inter = interpolate_vector(z_1, z_2, n_inter)

            # Generate interpolation images
            ims = gen_fn(z_inter, y_inter)

            for q in range(0, ims.shape[0]):
                interpolations[self.li * (n / 2): self.li * (n / 2) + self.li, self.li * q:self.li * q + self.li,
                :] = deprocess_image(
                    ims[q, :, :, :], self.li, self.nc)

        # Plot interpolation
        fig, ax = plt.subplots()

        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_title("Interpolation between 2 Images")

        if self.nc == 1:
            plt.imshow(np.squeeze(interpolations), cmap='gray')
        else:
            plt.imshow(interpolations)

        fig.savefig(self.base + '/images/interpolation.png', bbox_inches='tight')
        plt.close(fig)


