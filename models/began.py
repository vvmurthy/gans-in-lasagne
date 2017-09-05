import os
import time

import theano
import theano.tensor as T
import lasagne

from utils.iters import *
from utils.graphs_unconditional import *
from utils.z_vars import *
from utils.layers import *
from math import sqrt
import matplotlib.pyplot as plt
from utils.gen_utils import *


class BEGAN:

    def __init__(self, folder_name, dataset, **kwargs):

        # set hyperparameters
        self.start_lr = kwargs.get('lr', 0.0001)
        self.lr = self.start_lr
        self.num_epochs = kwargs.get('num_epochs', 25)
        self.bz = kwargs.get('bz', 64)
        self.num_hidden = kwargs.get('num_hidden', 64)
        self.folder_name = folder_name
        self.z_var = kwargs.get('z_var', z_var_uniform)
        self.gamma = kwargs.get('gamma', 0.7)
        self.num_filters = kwargs.get('num_filters', 64)
        self.k_t = kwargs.get('k_t', 0)
        self.offset = kwargs.get('offset', 5)
        self.num_examples = kwargs.get('num_examples', 10)
        self.seed = kwargs.get('seed', 115)
        self.n_inter = kwargs.get('n_inter', 10)

        # set variables from dataset
        self.dataset = dataset
        self.X_files_train = dataset.X_files_train
        self.li = dataset.li
        self.nc = dataset.nc
        self.images_in_mem = dataset.images_in_mem

        # Perform variable checks
        if 2 ** int(np.log2(self.li)) != self.li:
            raise ValueError(
                "Image length for BEGAN must be power of 2"
            )

        if int(sqrt(self.num_hidden)) != 2 ** int(np.log2(sqrt(self.num_hidden))):
            raise ValueError(
                "Number of hidden units must be an even power of 2"
            )

        self.base = os.getcwd() + '/' + folder_name + '/'
        if not os.path.isdir(self.base):
            os.mkdir(self.base)
            os.mkdir(self.base + 'models/')
            os.mkdir(self.base + 'images/')
            os.mkdir(self.base + 'stats/')

    def build_train_fns(self):

        def build_generator(li, num_filters, num_hidden, offset):

            z_var = T.fmatrix('z_var')
            generator = {}
            details = [['Layer Name', 'Dims in', 'shape of layer', 'Dims out']]

            name = 'gen_z_var'
            input_shape = (None, num_hidden)
            generator[name] = lasagne.layers.InputLayer(shape=input_shape, input_var=z_var)
            output_dims = input_shape

            prev_name = name
            name = 'fc_encode'
            num_units = num_hidden * num_filters
            generator[name] = lasagne.layers.DenseLayer(generator[prev_name], num_units, nonlinearity=None)
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(generator[name])
            details.append([name, str(prev_output_dims), str((product(prev_output_dims[1:]), num_units)),
                            str(output_dims)])

            # Reshape to batchsize x num_filters x 8 x 8
            prev_name = name
            name = 'gen_reshape'
            new_size = int(sqrt(num_hidden))
            generator[name] = lasagne.layers.ReshapeLayer(generator[prev_name], ([0], num_filters, new_size, new_size))
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(generator[name])
            details.append([name, str(prev_output_dims), str('reshape to 4D'),
                            str(output_dims)])

            # two convolutions + upscale in each repeat
            filter_size = 3
            scale_factor = 2

            # calculates repeats - we assume li of 2^x where x is a positive integer - (e.g. 64, 128, 32, etc)
            repeat_num = int(np.log2(np.array(li)) - np.log2(new_size))

            for n in range(0, repeat_num + 1):

                if n >= offset:
                    filters = num_filters * (n + 1 - offset)
                else:
                    filters = num_filters

                prev_name = name
                name = 'gen_conv' + str(n)
                prev_num_filters = lasagne.layers.get_output_shape(generator[prev_name])[1]
                generator[name] = lasagne.layers.Conv2DLayer(generator[prev_name], filters,
                                                             filter_size, stride=1, pad='same',
                                                             nonlinearity=lasagne.nonlinearities.elu)
                prev_output_dims = output_dims
                output_dims = lasagne.layers.get_output_shape(generator[name])
                details.append([name, str(prev_output_dims), str((filters, prev_num_filters, filter_size, filter_size)),
                                str(output_dims)])

                prev_name = name
                name = 'gen_conv' + str(n) + '_' + str(2)
                prev_num_filters = lasagne.layers.get_output_shape(generator[prev_name])[1]
                generator[name] = lasagne.layers.Conv2DLayer(generator[prev_name], filters,
                                                             filter_size, stride=1, pad='same',
                                                             nonlinearity=lasagne.nonlinearities.elu)
                prev_output_dims = output_dims
                output_dims = lasagne.layers.get_output_shape(generator[name])
                details.append([name, str(prev_output_dims), str((filters, prev_num_filters, filter_size, filter_size)),
                                str(output_dims)])

                if n < repeat_num:
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
                                                         filter_size, stride=1, pad='same',
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
            print("Number of parameters " + str(lasagne.layers.count_params(generator['gen_out'])))

            return generator, z_var

        def build_discriminator(li, num_filters, num_hidden, offset):

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
            name = 'conv0_0'
            prev_num_filters = lasagne.layers.get_output_shape(discriminator[prev_name])[1]
            discriminator[name] = lasagne.layers.Conv2DLayer(discriminator[prev_name], num_filters,
                                                             filter_size, stride=1, pad='same', nonlinearity=elu)
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(discriminator[name])
            details.append([name, str(prev_output_dims), str((num_filters, prev_num_filters, filter_size, filter_size)),
                            str(output_dims)])

            repeat_num = int(np.log2(np.array([li])) - np.log2(int(sqrt(num_hidden))))

            for n in range(0, repeat_num):

                if n >= offset:
                    filters = (n - offset + 1) * num_filters
                else:
                    filters = num_filters

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
                                                                 filter_size, stride=1, pad='same', nonlinearity=elu)
                prev_output_dims = output_dims
                output_dims = lasagne.layers.get_output_shape(discriminator[name])
                details.append([name, str(prev_output_dims), str((filters, prev_num_filters, filter_size, filter_size)),
                                str(output_dims)])

                if n < 1 + repeat_num:
                    prev_name = name
                    name = 'conv' + str(n)
                    prev_num_filters = lasagne.layers.get_output_shape(discriminator[prev_name])[1]
                    discriminator[name] = lasagne.layers.Conv2DLayer(discriminator[prev_name], filters,
                                                                     filter_size, stride=2, pad='same',
                                                                     nonlinearity=elu)
                    prev_output_dims = output_dims
                    output_dims = lasagne.layers.get_output_shape(discriminator[name])
                    details.append(
                        [name, str(prev_output_dims), str((filters, prev_num_filters, filter_size, filter_size)),
                         str(output_dims)])

            # Fully connected layers

            prev_name = name
            name = 'fc_encode'
            num_units = num_hidden
            discriminator[name] = lasagne.layers.DenseLayer(discriminator[prev_name], num_units, nonlinearity=None)
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(discriminator[name])
            details.append([name, str(prev_output_dims), str((product(prev_output_dims[1:]), num_units)),
                            str(output_dims)])

            prev_name = name
            name = 'fc_decode'
            num_units = num_hidden * num_filters
            discriminator[name] = lasagne.layers.DenseLayer(discriminator[prev_name], num_units, nonlinearity=None)
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(discriminator[name])
            details.append([name, str(prev_output_dims), str((product(prev_output_dims[1:]), num_units)),
                            str(output_dims)])

            prev_name = name
            name = 'reshape'
            new_size = int(sqrt(num_hidden))
            discriminator[name] = lasagne.layers.ReshapeLayer(discriminator[prev_name],
                                                              ([0], num_filters, new_size, new_size))
            prev_output_dims = output_dims
            output_dims = lasagne.layers.get_output_shape(discriminator[name])
            details.append([name, str(prev_output_dims), str('reshape to 4D'),
                            str(output_dims)])

            scale_factor = 2

            for n in range(0, repeat_num + 1):

                if n >= offset:
                    filters = num_filters * (n + 1 - offset)
                else:
                    filters = num_filters

                prev_name = name
                name = 'decode_conv' + str(n)
                prev_output_dims = output_dims
                prev_num_filters = prev_output_dims[1]
                discriminator[name] = lasagne.layers.Conv2DLayer(discriminator[prev_name], filters,
                                                                 filter_size, stride=1, pad='same', nonlinearity=elu)
                output_dims = lasagne.layers.get_output_shape(discriminator[name])
                details.append([name, str(prev_output_dims), str((filters, prev_num_filters, filter_size, filter_size)),
                                str(output_dims)])

                prev_name = name
                name = 'decode_conv' + str(n) + '_2'
                prev_num_filters = lasagne.layers.get_output_shape(discriminator[prev_name])[1]
                discriminator[name] = lasagne.layers.Conv2DLayer(discriminator[prev_name], filters,
                                                                 filter_size, stride=1, pad='same', nonlinearity=elu)
                prev_output_dims = output_dims
                output_dims = lasagne.layers.get_output_shape(discriminator[name])
                details.append([name, str(prev_output_dims), str((filters, prev_num_filters, filter_size, filter_size)),
                                str(output_dims)])

                if n < repeat_num:
                    prev_name = name
                    name = 'upscale_' + str(n)
                    discriminator = nearest_neighbor(discriminator, name, prev_name, output_dims[2], scale_factor)
                    output_dims = (None, filters, scale_factor * output_dims[2], scale_factor * output_dims[2])
                    details.append([name, str(prev_output_dims), str('2x upscale'),
                                    str(output_dims)])

                # reset number of filters
                filters = num_filters

            prev_name = name
            name = 'out'
            num_filters = 3
            discriminator[name] = lasagne.layers.Conv2DLayer(discriminator[prev_name], num_filters,
                                                             filter_size, stride=1, pad='same',
                                                             nonlinearity=lasagne.nonlinearities.tanh)
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

        # defines variables
        print("Building model and compiling functions...")

        # Builds discriminator and generator
        k_t = T.fscalar('k_t')
        gamma = theano.compile.shared(self.gamma)
        discriminator, input_var = build_discriminator(self.li, self.num_filters, self.num_hidden, self.offset)
        generator, z_var = build_generator(self.li, self.num_filters, self.num_hidden, self.offset)

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
        dis_train_fn = theano.function([input_var, z_var, lr, k_t],
                                       [real_out, k_tp, discriminator_loss, convergence_measure],
                                       updates=dis_updates)

        # generates images for validation
        gen_fn = theano.function([z_var], lasagne.layers.get_output(generator['gen_out'], deterministic=True))
        print("...Done")
        return generator, discriminator, gen_train_fn, gen_fn, dis_train_fn

    def train(self):
        # Make training functions
        print("Making Training Functions...")
        generator, discriminator, gen_train_fn, gen_fn, dis_train_fn = self.build_train_fns()

        # Load in params if training incomplete
        try:

            # Load training statistics
            start_epoch = np.load(self.base + 'stats/epoch.npy')[0]
            gen_train_err = np.load(self.base + 'stats/gen_train_err.npy')
            dis_train_err_real = np.load(self.base + 'stats/dis_train_err_real.npy')
            convergence_measure = np.load(self.base + 'stats/convergence_measure.npy')

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
            convergence_measure = np.zeros((self.num_epochs)).astype(np.float32)
            print("...Loaded models")

        print("Starting BEGAN Training...")
        for epoch in range(start_epoch, self.num_epochs):
            start_time = time.time()

            num_batches = 0

            # Load specified amount of images into memory at once
            for X_files_mem in iterate_membatches_unconditional(self.X_files_train,
                                                                self.images_in_mem,
                                                                self.dataset.load_files,
                                                                self.li,
                                                                shuffle=True):

                for inputs in iterate_minibatches_unconditional(X_files_mem, self.bz, shuffle=True):
                    # Create noise vector
                    noise = self.z_var(self.bz, self.num_hidden)

                    # Train the generator
                    fake_out, ims, gen_train_err_epoch = gen_train_fn(noise, self.lr)
                    gen_train_err[epoch] += gen_train_err_epoch

                    # Train the discriminator
                    #   + real_out - predictions on real images
                    real_out, k_t, dis_train_err_real_epoch, convergence_measure_epoch = dis_train_fn(inputs,
                                                                                                      noise,
                                                                                                      self.lr,
                                                                                                      self.k_t)
                    k_t = np.float32(k_t)
                    dis_train_err_real[epoch] += dis_train_err_real_epoch
                    convergence_measure[epoch] += convergence_measure_epoch

                    num_batches += 1

            # Display training stats
            print("Epoch {} of {} took {:.3f} minutes".format(epoch + 1, self.num_epochs,
                                                              (time.time() - start_time) / np.float32(60)))
            print("  Generator Error:\t\t{}".format(gen_train_err[epoch] / num_batches))
            print("  Discriminator Error:\t\t{}".format(dis_train_err_real[epoch] / num_batches))
            print("  Convergence Measure:\t\t{}".format(convergence_measure[epoch] / num_batches))

            # Save stats + models
            np.save(self.base + 'stats/epoch.npy', np.array([epoch]))
            np.save(self.base + 'stats/gen_train_err.npy', gen_train_err)
            np.save(self.base + 'stats/dis_train_err_real.npy', dis_train_err_real)
            np.save(self.base + 'stats/convergence_measure.npy', convergence_measure)
            np.savez(self.base + 'models/generator_epoch' + str(epoch) + '.npz',
                     *lasagne.layers.get_all_param_values(generator['gen_out']))
            np.savez(self.base + 'models/discriminator_epoch' + str(epoch) + '.npz',
                     *lasagne.layers.get_all_param_values(discriminator['out']))

            if epoch >= 1 and convergence_measure[epoch] - convergence_measure[epoch - 1] > -20:
                print("Decaying Learning rate: Epoch " + str(epoch))
                self.lr = self.lr / 2

            # Create num_examples_row ** 2 generated images
            max_ims = int(self.num_examples ** 2 / self.bz) + 1
            val_ims = np.zeros((max_ims * self.bz, self.nc, self.li, self.li))
            for st in range(0, max_ims):
                noise = self.z_var(self.bz, self.num_hidden)
                val_ims[self.bz * st: self.bz * st + self.bz] = gen_fn(noise)

            show_examples_unlabeled(val_ims, self.num_examples, self.li, self.nc, epoch,
                                    self.base + 'images/epoch' + str(epoch) + '.png')

        # Make graph with training statistics
        show_training_stats_graph(gen_train_err, dis_train_err_real, convergence_measure,
                                  self.num_epochs, self.base + 'stats/stats_graph.png',
                                  'discriminator error', 'convergence measure')

        # Save final models
        np.savez(self.base + 'models/generator_final.npz', *lasagne.layers.get_all_param_values(generator['gen_out']))
        np.savez(self.base + 'models/discriminator_final.npz', *lasagne.layers.get_all_param_values(discriminator['out']))

        print("...BEGAN Training Complete")

    def test(self):
        # seed indices generation
        np.random.seed(self.seed)

        # Check that models do exist
        assert (os.path.isfile(self.base + '/models/generator_final.npz')), \
            ("Generator in " + self.base + " Does not exist")
        assert (os.path.isfile(self.base + '/models/discriminator_final.npz')), \
            ("Discriminator in " + self.base + " Does not exist")

        # Build GAN + Encoder
        generator, discriminator, gen_train_fn, gen_fn, dis_train_fn = self.build_train_fns()

        # Set params
        with np.load(self.base + '/models/generator_final.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(generator['gen_out'], param_values)

        with np.load(self.base + '/models/discriminator_final.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(discriminator['out'], param_values)

        # Generate several random images
        max_ims = int(self.num_examples ** 2 / self.bz) + 1
        val_ims = np.zeros((max_ims * self.bz, self.nc, self.li, self.li))
        for st in range(0, max_ims):
            noise = self.z_var(self.bz, self.num_hidden)
            val_ims[self.bz * st: self.bz * st + self.bz] = gen_fn(noise)

        show_examples_unlabeled(val_ims, self.num_examples, self.li, self.nc, None,
                                self.base + '/images/examples.png')

        interpolations = np.zeros((self.li * self.num_examples / 2, self.li * self.n_inter + self.li, self.nc)).astype(np.float32)
        for n in range(0, self.num_examples, 2):

            z_1 = self.z_var(1, self.num_hidden)[0]
            z_2 = self.z_var(1, self.num_hidden)[0]

            # Interpolate z
            z_inter = interpolate_vector(z_1, z_2, self.n_inter)

            # Generate interpolation images
            ims = gen_fn(z_inter)

            for q in range(0, ims.shape[0]):
                interpolations[self.li * (n / 2): self.li * (n / 2) + self.li,
                self.li * q : self.li * q + self.li, :] = deprocess_image(
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
