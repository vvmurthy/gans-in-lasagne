import os
import time

import lasagne
import numpy as np

from archive.utils import iterate_minibatches_conditional, iterate_membatches
from archive.utils import randomize_y, show_reconstructions
from archive.utils import show_training_stats_graph, show_examples, show_encoder_stats_graph
from config import config
from models.build_encoders import build_encoder_z, build_encoder_y
from models.build_gans import make_train_fns


# Soon to be renamed train_icgan - train ICGAN model
def train():

    # Set hyperparameters
    configuration = config()
    start_lr = configuration['lr']
    images_in_mem = configuration['images_in_mem']
    lr = start_lr
    num_epochs = configuration['num_epochs']
    bz = configuration['bz']
    li = configuration['li']
    nc = configuration['nc']
    lab_ln = configuration['lab_ln']
    folder_name = configuration['folder_name']

    # Set variables from dataset
    X_files_train = configuration['X_files_train']
    y_train = configuration['y_train']
    X_files_val = configuration['X_files_val']
    y_val = configuration['y_val']
    labels = configuration['labels']

    # Set file loader, iterator
    batch_iterator = iterate_minibatches_conditional
    dataset_loader = configuration['dataset_loader']

    # Make folders for storing models
    base = os.getcwd() + '/' + folder_name + '/'
    if not os.path.isdir(base):
        os.mkdir(base)
        os.mkdir(base + 'models/')
        os.mkdir(base + 'images/')
        os.mkdir(base + 'stats/')

    # Make training functions
    print("Making Training Functions...")
    generator, discriminator, gen_train_fn, gen_fn, dis_train_fn = make_train_fns(bz, li, nc, lab_ln)

    # Load in params if training incomplete
    try:

        # Load training statistics
        start_epoch = np.load(base + 'stats/epoch.npy')[0]
        gen_train_err = np.load(base + 'stats/gen_train_err.npy')
        dis_train_err_real = np.load(base + 'stats/dis_train_err_real.npy')
        dis_train_err_fake = np.load(base + 'stats/dis_train_err_fake.npy')

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
        dis_train_err_fake = np.zeros((num_epochs)).astype(np.float32)
        print("...Loaded models")

    print("Starting cGAN Training...")
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        # Train cGAN
        num_batches = 0
        X_files_train = X_files_train[0: (X_files_train.shape[0] / bz) * bz]
        y_train = y_train[0: (y_train.shape[0] / bz) * bz, :]

        # Load 64 * batches_in_mem images onto the memory at one time
        indices = np.arange(X_files_train.shape[0])
        np.random.shuffle(indices)
        for X_files_mem in iterate_membatches(X_files_train, images_in_mem, dataset_loader, li, shuffle=True):

            for batch in batch_iterator(X_files_mem, y_train, bz, shuffle=True):
                inputs, targets = batch

                # expands dims for training (generator + discriminator expect 3D input)
                targets = np.expand_dims(targets, 2)

                # Create noise vector
                noise = np.array(np.random.uniform(-1, 1, (bz, 100))).astype(np.float32)
                y_fake = randomize_y(targets)

                # Train the generator
                fake_out, ims, gen_train_err_epoch = gen_train_fn(noise, targets, lr)
                gen_train_err[epoch] += gen_train_err_epoch

                # Train the discriminator
                #   + real_out - predictions on real images + matching y vectors
                #   + real_out_yfake - predictions on real_images + fake y vectors
                real_out, real_out_yfake, dis_train_err_real_epoch, dis_train_err_fake_epoch = dis_train_fn(inputs, noise,
                                                                                                            targets, y_fake,
                                                                                                            lr)
                dis_train_err_real[epoch] += dis_train_err_real_epoch
                dis_train_err_fake[epoch] += dis_train_err_fake_epoch

                num_batches += 1

        # Display training stats
        print("Epoch {} of {} took {:.3f} minutes".format(epoch + 1, num_epochs, (time.time() - start_time) / np.float32(60)))
        print("  Generator Accuracy:\t\t{}".format(gen_train_err[epoch] / num_batches))
        print("  Discriminator Accuracy on real ims:\t\t{}".format(dis_train_err_real[epoch] / num_batches))
        print("  Discriminator Accuracy on fake ims:\t\t{}".format(dis_train_err_fake[epoch] / num_batches))

        # Save stats + models
        np.save(base + 'stats/epoch.npy', np.array([epoch]))
        np.save(base + 'stats/gen_train_err.npy', gen_train_err)
        np.save(base + 'stats/dis_train_err_real.npy', dis_train_err_real)
        np.save(base + 'stats/dis_train_err_fake.npy', dis_train_err_fake)
        np.savez(base + 'models/generator_epoch' + str(epoch) + '.npz',
                 *lasagne.layers.get_all_param_values(generator['gen_out']))
        np.savez(base + 'models/discriminator_epoch' + str(epoch) + '.npz',
                 *lasagne.layers.get_all_param_values(discriminator['out']))

        # Decay the lr
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            lr = start_lr * 2 * (1 - progress)

        # Do a pass over first 10 sets of 64 y vectors from validation set every epoch, show example images
        # note many example images are repeated between categories
        sets = 10
        val_ims = np.zeros((bz * sets, 3, li, li))
        for st in range(0, sets):
            noise = np.array(np.random.uniform(-1, 1, (bz, 100))).astype(np.float32)
            targets = np.expand_dims(y_val[bz * st: bz * st + bz], 2)
            val_ims[bz * st: bz * st + bz] = gen_fn(noise, targets)

        show_examples(val_ims, y_val[0:bz * sets], labels, li, epoch, base + 'images/epoch' + str(epoch) + '.png')

    # Make graph with training statistics
    show_training_stats_graph(gen_train_err, dis_train_err_real, dis_train_err_fake, num_epochs,
                              base + 'stats/stats_graph.png')

    # Save final models
    np.savez(base + 'models/generator_final.npz', *lasagne.layers.get_all_param_values(generator['gen_out']))
    np.savez(base + 'models/discriminator_final.npz', *lasagne.layers.get_all_param_values(discriminator['out']))

    print("...cGAN Training Complete")
    print("Building Encoder Models...")

    # Build Encoder
    encoder_z, encoder_z_train, encoder_z_test = build_encoder_z(li, nc, lr)
    encoder_y, encoder_y_train, encoder_y_test = build_encoder_y(li, nc, lab_ln, lr)

    # Load in params if partial training
    try:

        # Load training statistics
        start_epoch = np.load(base + 'stats/encoder_epoch.npy')[0]
        encoder_z_loss = np.load(base + 'stats/encoder_z_loss.npy')
        encoder_y_loss = np.load(base + 'stats/encoder_y_loss.npy')

        # Load models
        with np.load(base + 'encoder_z_epoch' + str(start_epoch) + '.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(encoder_z, param_values)

        with np.load(base + 'encoder_y_epoch' + str(start_epoch) + '.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(encoder_y, param_values)

        start_epoch += 1
        print("...Loaded partially trained models")
    except IOError:
        start_epoch = 0
        encoder_y_loss = np.zeros((num_epochs)).astype(np.float32)
        encoder_z_loss = np.zeros((num_epochs)).astype(np.float32)
        print("...Built models")

    # Make some generated samples for training encoder
    # Train encoder as well
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        num_batches = 0
        for batch in batch_iterator(X_files_train, y_train, bz, shuffle=True):
            _, targets = batch
            gen_targets = np.expand_dims(targets, 2)
            noise = np.array(np.random.uniform(-1, 1, (bz, 100))).astype(np.float32)
            gen_images = gen_fn(noise, gen_targets)
            encoder_z_loss[epoch] += encoder_z_train(gen_images, noise)
            encoder_y_loss[epoch] += encoder_y_train(gen_images, targets)
            num_batches += 1

        # Print training stats
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  Encoder Z loss:\t\t{:.3f}".format(encoder_z_loss[epoch] / num_batches))
        print("  Encoder y loss:\t\t{:.3f}".format(encoder_y_loss[epoch] / num_batches))

        # Show sample reconstructions from y val
        num_examples = 10
        ims = dataset_loader(X_files_val[0: num_examples], num_examples, li)
        z = encoder_z_test(ims)[0]
        y = np.expand_dims(encoder_y_test(ims)[0], 2)
        reconstructions = gen_fn(z, y)
        show_reconstructions(ims, reconstructions, li, epoch, base + 'images/reconstructions_epoch' + str(epoch) + '.png')


        # Save training stats and intermediate models
        np.save(base + 'stats/encoder_epoch.npy', np.array([epoch]))
        np.save(base + 'stats/encoder_z_loss.npy', encoder_z_loss)
        np.save(base + 'stats/encoder_y_loss.npy', encoder_y_loss)
        np.savez(base + 'models/encoder_z_epoch' + str(epoch) + '.npz',
                 *lasagne.layers.get_all_param_values(encoder_z['out']))
        np.savez(base + 'models/encoder_y_epoch' + str(epoch) + '.npz',
                 *lasagne.layers.get_all_param_values(encoder_y['out']))

    # Save encoder (final)
    np.savez(base + 'models/encoder_z_final.npz', *lasagne.layers.get_all_param_values(encoder_z['out']))
    np.savez(base + 'models/encoder_y_final.npz', *lasagne.layers.get_all_param_values(encoder_y['out']))

    # Make training stats graph
    show_encoder_stats_graph(encoder_z_loss, encoder_y_loss, num_epochs, base + 'stats/encoder_stats.png')

train()
