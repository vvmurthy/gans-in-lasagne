import os
import time

import lasagne
import numpy as np

from utils.iters import iterate_minibatches_unconditional, iterate_membatches_unconditional
from utils.graphs_unconditional import show_training_stats_graph, show_examples_unlabeled
from models.build_began import make_train_fns


# train BEGAN model
def train_began(configuration):

    # Set hyperparameters
    start_lr = configuration['lr']
    images_in_mem = configuration['images_in_mem']
    lr = start_lr
    num_epochs = configuration['num_epochs']
    bz = configuration['bz']
    li = configuration['li']
    nc = configuration['nc']
    folder_name = configuration['folder_name']

    # Set variables from dataset
    X_files_train = configuration['X_files_train']

    # Set file loader, iterator
    batch_iterator = iterate_minibatches_unconditional
    dataset_loader = configuration['dataset_loader']

    # Set began-specific variables
    gamma = configuration['gamma']
    num_filters = configuration['num_filters']
    k_t = np.float32(configuration['k_t'])
    
    # Make folders for storing models
    base = os.getcwd() + '/' + folder_name + '/'
    if not os.path.isdir(base):
        os.mkdir(base)
        os.mkdir(base + 'models/')
        os.mkdir(base + 'images/')
        os.mkdir(base + 'stats/')

    # Make training functions
    print("Making Training Functions...")
    generator, discriminator, gen_train_fn, gen_fn, dis_train_fn = make_train_fns(li, gamma, num_filters)

    # Load in params if training incomplete
    try:

        # Load training statistics
        start_epoch = np.load(base + 'stats/epoch.npy')[0]
        gen_train_err = np.load(base + 'stats/gen_train_err.npy')
        dis_train_err_real = np.load(base + 'stats/dis_train_err_real.npy')
        convergence_measure = np.load(base + 'stats/convergence_measure.npy')

        # Load models
        with np.load(base + 'models/generator_epoch' + str(start_epoch) + '.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(generator['gen_out'], param_values)

        with np.load(base + 'models/discriminator_epoch' + str(start_epoch) + '.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(discriminator['out'], param_values)

        start_epoch += 1
        print("...Loaded previous models")
    except IOError:
        start_epoch = 0
        gen_train_err = np.zeros((num_epochs)).astype(np.float32)
        dis_train_err_real = np.zeros((num_epochs)).astype(np.float32)
        convergence_measure = np.zeros((num_epochs)).astype(np.float32)
        print("...Loaded models")

    print("Starting BEGAN Training...")
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        num_batches = 0

        # Load specified amount of images into memory at once
        for X_files_mem in iterate_membatches_unconditional(X_files_train, images_in_mem, dataset_loader, li, shuffle=True):

            for inputs in batch_iterator(X_files_mem, bz, shuffle=True):

                print(num_batches)

                # Create noise vector
                noise = np.array(np.random.uniform(-1, 1, (bz, 8*8))).astype(np.float32)

                # Train the generator
                fake_out, ims, gen_train_err_epoch = gen_train_fn(noise, lr)
                gen_train_err[epoch] += gen_train_err_epoch

                # Train the discriminator
                #   + real_out - predictions on real images
                real_out, k_t, dis_train_err_real_epoch, convergence_measure_epoch = dis_train_fn(inputs, noise, lr, k_t)
                k_t = np.float32(k_t)                
                dis_train_err_real[epoch] += dis_train_err_real_epoch
                convergence_measure[epoch] += convergence_measure_epoch

                num_batches += 1

        # Display training stats
        print("Epoch {} of {} took {:.3f} minutes".format(epoch + 1, num_epochs, (time.time() - start_time) / np.float32(60)))
        print("  Generator Error:\t\t{}".format(gen_train_err[epoch] / num_batches))
        print("  Discriminator Error:\t\t{}".format(dis_train_err_real[epoch] / num_batches))
        print("  Convergence Measure:\t\t{}".format(convergence_measure[epoch] / num_batches))

        # Save stats + models
        np.save(base + 'stats/epoch.npy', np.array([epoch]))
        np.save(base + 'stats/gen_train_err.npy', gen_train_err)
        np.save(base + 'stats/dis_train_err_real.npy', dis_train_err_real)
        np.save(base + 'stats/convergence_measure.npy', convergence_measure)
        np.savez(base + 'models/generator_epoch' + str(epoch) + '.npz',
                 *lasagne.layers.get_all_param_values(generator['gen_out']))
        np.savez(base + 'models/discriminator_epoch' + str(epoch) + '.npz',
                 *lasagne.layers.get_all_param_values(discriminator['out']))

        # Decay the lr
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            lr = start_lr * 2 * (1 - progress)

        # Create 100 example generated images after each epoch
        sets = 100
        val_ims = np.zeros((bz * sets, nc, li, li))
        for st in range(0, sets):
            noise = np.array(np.random.uniform(-1, 1, (bz, 8*8))).astype(np.float32)
            val_ims[bz * st: bz * st + bz] = gen_fn(noise)

        show_examples_unlabeled(val_ims, li, nc, epoch, base + 'images/epoch' + str(epoch) + '.png')

    # Make graph with training statistics
    show_training_stats_graph(gen_train_err, dis_train_err_real, convergence_measure, num_epochs,
                              base + 'stats/stats_graph.png')

    # Save final models
    np.savez(base + 'models/generator_final.npz', *lasagne.layers.get_all_param_values(generator['gen_out']))
    np.savez(base + 'models/discriminator_final.npz', *lasagne.layers.get_all_param_values(discriminator['out']))

    print("...BEGAN Training Complete")
