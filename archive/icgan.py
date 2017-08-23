import os
import time

import lasagne
import matplotlib.pyplot as plt
import numpy as np

from archive.utils import randomize_y
from archive.utils import show_training_stats_graph, show_examples, show_encoder_stats_graph, modify_y, \
    interpolate_vector
from datasets.celeba import load_xy, load_files, iterate_minibatches
from models.build_encoders import build_encoder_z, build_encoder_y
from models.build_gans import make_train_fns


def train(base):

    # Load dataset- filenames for images, all y attributes
    print("Loading Dataset...")
    X_files_train, y_train, X_files_val, y_val, X_files_test, y_test, labels = load_xy()

    # Make training functions
    print("Making Training Functions...")
    generator, discriminator, gen_train_fn, gen_fn, dis_train_fn = make_train_fns()

    # Set hyperparameters
    start_lr = 0.0002
    lr = start_lr
    num_epochs = 25
    batchsize = 64

    # Load in params if training incomplete
    try:

        # Load training statistics
        start_epoch = np.load(base + 'stats/epoch.npy')[0]
        gen_train_err = np.load(base + 'stats/gen_train_err.npy')
        dis_train_err_real = np.load(base + 'stats/dis_train_err_real.npy')
        dis_train_err_fake = np.load(base + 'stats/dis_train_err_fake.npy')

        # Load models
        with np.load(base + 'generator_epoch' + str(start_epoch) + '.npz') as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(generator, param_values)

        with np.load(base + 'discriminator_epoch' + str(start_epoch) + '.npz') as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(discriminator, param_values)

        start_epoch += 1
        print("Loaded previous models...")
    except IOError:
        start_epoch = 0
        gen_train_err = np.zeros((num_epochs)).astype(np.float32)
        dis_train_err_real = np.zeros((num_epochs)).astype(np.float32)
        dis_train_err_fake = np.zeros((num_epochs)).astype(np.float32)

    print("Starting cGAN Training...")
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        # Train cGAN
        num_batches = 0
        X_files_train = X_files_train[0 : (X_files_train.shape[0] / batchsize) * batchsize]
        y_train = y_train[0 : (y_train.shape[0] / batchsize) * batchsize, :]
        for batch in iterate_minibatches(X_files_train, y_train, batchsize, shuffle=True):
            print(num_batches)
            files, targets = batch
            
            #expands dims for training (see build_gans)
            targets = np.expand_dims(targets, 2)

            # Load in images, create noise vector
            inputs = load_files(files, batchsize)
            noise = np.array(np.random.uniform(-1, 1, (64, 100))).astype(np.float32)
            y_fake = randomize_y(targets)
            
            # Train the generator
            fake_out, ims, gen_train_err_epoch = gen_train_fn(noise, targets, lr)
            gen_train_err[epoch] += gen_train_err_epoch
            
            # Train the discriminator
            real_out, real_out_yfake, dis_train_err_real_epoch, dis_train_err_fake_epoch = dis_train_fn(inputs, noise, targets, y_fake, lr)
            dis_train_err_real[epoch] += dis_train_err_real_epoch
            dis_train_err_fake[epoch] += dis_train_err_fake_epoch
            
            num_batches += 1

        # Display training stats
        print("Epoch {} of {} took {:.3f} minutes".format(epoch + 1, num_epochs, (time.time() - start_time)/np.float32(60)))
        print("  Generator Accuracy:\t\t{}".format(gen_train_err[epoch] / num_batches))
        print("  Discriminator Accuracy on real ims:\t\t{}".format(dis_train_err_real[epoch] / num_batches))
        print("  Discriminator Accuracy on fake ims:\t\t{}".format(dis_train_err_fake[epoch] / num_batches))

        # Save stats + models
        np.save(base + 'stats/gen_train_err.npy', gen_train_err)
        np.save(base + 'stats/dis_train_err_real.npy', dis_train_err_real)
        np.save(base + 'stats/dis_train_err_fake.npy',dis_train_err_fake)
        np.savez(base + 'models/generator_epoch' + str(epoch) + '.npz', *lasagne.layers.get_all_param_values(generator['gen_out']))
        np.savez(base + 'models/discriminator_epoch' + str(epoch) + '.npz', *lasagne.layers.get_all_param_values(discriminator['out']))

        # Decay the lr
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            lr = start_lr * 2 * (1 - progress)

        # Do a pass over first n sets of 64 y vectors from validation set every epoch, show example images
        sets = 10
        val_ims = np.zeros((64*sets, 3, 64, 64))        
        for st in range(0, sets):
            noise = np.array(np.random.uniform(-1, 1, (64, 100))).astype(np.float32)
            targets = np.expand_dims(y_val[64*st : 64*st + 64], 2)
            val_ims[64*st : 64*st + 64] = gen_fn(noise, targets)
            
        show_examples(val_ims, y_val[0:64*sets], labels, base + 'images/epoch' + str(epoch) + '.png')

    # Make graph with training statistics
    show_training_stats_graph(gen_train_err, dis_train_err_real, dis_train_err_fake, num_epochs, base + 'stats/stats_graph.png')

    # Save models final
    np.savez(base + 'models/generator_final.npz', *lasagne.layers.get_all_param_values(generator['gen_out']))
    np.savez(base + 'models/discriminator_final.npz', *lasagne.layers.get_all_param_values(discriminator['out']))

    # Build Encoder
    encoder_z, encoder_z_train, encoder_z_test = build_encoder_z()
    encoder_y, encoder_y_train, encoder_y_test = build_encoder_y()

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
    except IOError:
        start_epoch = 0
        encoder_y_loss = np.zeros((num_epochs)).astype(np.float32)
        encoder_z_loss = np.zeros((num_epochs)).astype(np.float32)

    # Make some generated samples for training encoder
    # Train encoder as well
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        for batch in iterate_minibatches(X_files_train, y_train, 64, shuffle=True):
            files, targets = batch
            inputs = load_files(files, 64)
            noise = np.array(np.random.uniform(-1, 1, (64, 100))).astype(np.float32)
            gen_images = gen_fn(noise, targets)
            encoder_z_loss[epoch] += encoder_z_train(gen_images, noise)
            encoder_y_loss[epoch] += encoder_y_train(gen_images, targets)

        # Print training stats
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  Encoder Z loss:\t\t{}".format(encoder_z_loss[epoch] / y_train.shape[0]))
        print("  Encoder y loss:\t\t{}".format(encoder_y_loss[epoch] / y_train.shape[0]))

        # Save training stats and intermediate models
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


# Benchmark on test data
def test(Train, name):

    # Make folders for storing models
    base = os.getcwd() + '/' + name + '/'
    if not os.path.isdir(base):
        os.mkdir(base)
        os.mkdir(base + 'models/')
        os.mkdir(base + 'images/')
        os.mkdir(base + 'stats/')

    if Train:
        train(base)

    # Build GAN + Encoder
    generator, discriminator, gen_train_fn, gen_fn, dis_train_fn = make_train_fns()
    encoder_z, encoder_z_train, encoder_z_test = build_encoder_z()
    encoder_y, encoder_y_train, encoder_y_test = build_encoder_y()

    # Set params
    with np.load(base + 'generator_final.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(generator, param_values)

    with np.load(base + 'discriminator_final.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(discriminator, param_values)

    with np.load(base + 'encoder_z_final.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(encoder_z, param_values)

    with np.load(base + 'encoder_y_final.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(encoder_y, param_values)


    num_people = 4 # use 4 people for each example figure (2 for swap)

    #Load dataset
    _, _, _, _, X_files_test, y_test, labels = load_xy()

    # Reconstruct + change attributes
    # Torch implementation gets running train batch norms std and mean, this uses fixed vals
    all_reconstructions = np.zeros((64*num_people, 64*19, 3)).astype(np.float32)
    for index in range(0, num_people):
        image = load_files(X_files_test[index],1)
        y_pred = encoder_y_test(image)
        z = encoder_z_test(image)

        # Duplicate Z 19 tiles
        z_permutations = np.zeros((19,100)).astype(np.float32)
        for n in range(0, 19):
            z_permutations[n,:] = z

        # Create y matrix
        y_permutations = modify_y(y_pred)

        # Generate images
        generated_ims = gen_fn(z_permutations, y_permutations)

        # Map reconstructions to main image
        for n in range(0, generated_ims.shape[0]):

            for chan in range(0, 3):
                all_reconstructions[64*index: 64*index + 64, 64*n,: 64*n + 64 , chan] = generated_ims[n, chan, :, : ]

    # Plot the reconstruction
    fig, ax = plt.subplots()

    ax.set_yticks(np.arange(0, 64 * num_people, 64) + (64 / 2), minor=False)
    ax.set_xticks(np.arange(0, 64 * len(labels) + 64, 64) + (64 / 2), minor=False)
    ax.invert_yaxis()

    ax.set_xlabel('Description')
    ax.set_ylabel('Person')
    ax.set_title("Sample Generated Images")

    ax.set_xticklabels(labels, rotation='vertical', minor=False)
    plt.imshow(all_reconstructions)

    fig.savefig(base + 'images/reconstructions.png')
    plt.close(fig)

    # Swap
    image_1 = load_files(X_files_test[0], 1)
    image_2 = load_files(X_files_test[1], 1)
    y_1 = encoder_y_test(image_1)
    z_1 = encoder_z_test(image_1)
    y_2 = encoder_y_test(image_2)
    z_2 = encoder_z_test(image_2)

    num_pairs = 1
    swap_image = np.zeros((64 * num_pairs * 2, 64 * 3, 3)).astype(np.float32)
    for chan in range(0, 3):
        swap_image[0:64, 0:64, chan] = image_1[chan, :, :]
        swap_image[64:64 + 64, 0:64, chan] = image_2[chan, :, :]

    # Swaps for first image
    z_matrix = np.zeros((2, 100)).astype(np.float32)
    y_matrix = np.zeros((2, 18)).astype(np.float32)
    for n in range(0, 2):
        z_matrix[n, :] = z_1

    y_matrix[0, :] = y_1
    y_matrix[1, :] = y_2

    ims = gen_fn(z_matrix, y_matrix)
    for n in range(1, 3):
        for chan in range(0, 3):
            swap_image[0:64, n*64:n*64 + 64, chan] = ims[n - 1, chan, :, :]

    # Swaps for second image
    z_matrix = np.zeros((2, 100)).astype(np.float32)
    y_matrix = np.zeros((2, 18)).astype(np.float32)
    for n in range(0, 2):
        z_matrix[n, :] = z_2

    y_matrix[0, :] = y_2
    y_matrix[1, :] = y_1

    ims = gen_fn(z_matrix, y_matrix)
    for n in range(1, 3):
        for chan in range(0, 3):
            swap_image[64:64 + 64, n * 64:n * 64 + 64, chan] = ims[n - 1, chan, :, :]

    # Plot the swapped images
    x_lab = ['Original', 'Reconstruction', 'Swapped y']
    fig, ax = plt.subplots()

    ax.set_yticks(np.arange(0, 64 * num_pairs * 2, 64) + (64 / 2), minor=False)
    ax.set_xticks(np.arange(0, 64 * 3, 64) + (64 / 2), minor=False)
    ax.invert_yaxis()

    ax.set_ylabel('Person')
    ax.set_title("Swapped Images")

    ax.set_xticklabels(x_lab, rotation='vertical', minor=False)
    plt.imshow(all_reconstructions)

    fig.savefig(base + 'images/swapped.png')
    plt.close(fig)

    # Interpolation
    image_1 = load_files(X_files_test[0], 1)
    image_2 = load_files(X_files_test[1], 1)
    y_1 = encoder_y_test(image_1)
    z_1 = encoder_z_test(image_1)
    y_2 = encoder_y_test(image_2)
    z_2 = encoder_z_test(image_2)

    # Interpolate y and z
    n_inter = 10
    y_inter = interpolate_vector(y_1, y_2, 10)
    z_inter = interpolate_vector(z_1, z_2, 10)
    interpolations = np.zeros((64, 64* n_inter + 64, 3)).astype(np.float32)

    # Generate interpolation images
    ims = gen_fn(z_inter, y_inter)

    for n in range(0, ims.shape[0]):
        for chan in range(0, 3):
            interpolations[:, 64*n:64*n + 64, chan] = ims[n, chan, :, :]

    # Plot interpolation
    fig, ax = plt.subplots()

    ax.set_yticks(np.arange(0, 64 * num_people, 1) + (64 / 2), minor=False)
    ax.set_xticks(np.arange(0, 64 * n_inter + 64, 64) + (64 / 2), minor=False)
    ax.invert_yaxis()

    ax.set_title("Interpolation between 2 people")

    plt.imshow(interpolations)

    fig.savefig(base + 'images/interpolation.png')
    plt.close(fig)

test(True, 'first_test')