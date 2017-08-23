import os

import lasagne
import matplotlib.pyplot as plt
import numpy as np

from archive.utils import modify_y, interpolate_vector
from config import config
from datasets.celeba import load_files
from models.build_encoders import build_encoder_z, build_encoder_y
from models.build_gans import make_train_fns


# Benchmark on test data
def test():

    # Configure data
    configuration = config()

    # Check that models do exist
    assert(os.path.isfile(configuration['folder_name'] + 'generator_final.npz')), \
        ("Generator in " + configuration['folder_name'] + " Does not exist")
    assert (os.path.isfile(configuration['folder_name'] + 'discriminator_final.npz')), \
        ("Discriminator in " + configuration['folder_name'] + " Does not exist")
    assert (os.path.isfile(configuration['folder_name'] + 'encoder_z_final.npz')), \
        ("Encoder Z in " + configuration['folder_name'] + " Does not exist")
    assert (os.path.isfile(configuration['folder_name'] + 'encoder_y_final.npz')), \
        ("Encoder y in " + configuration['folder_name'] + " Does not exist")

    # Set hyperparameters
    li = configuration['li']
    nc = configuration['nc']
    lr = configuration['lr']
    bz = configuration['bz']
    lab_ln = configuration['lab_ln']
    folder_name = configuration['folder_name']

    # Set dataset
    X_files_test = configuration['X_files_test']
    y_test = configuration['y_test']
    labels = configuration['labels']

    # Build GAN + Encoder
    generator, discriminator, gen_train_fn, gen_fn, dis_train_fn = make_train_fns(bz, li, nc, lab_ln)
    encoder_z, encoder_z_train, encoder_z_test = build_encoder_z(li, nc, lr)
    encoder_y, encoder_y_train, encoder_y_test = build_encoder_y(li, nc, lab_ln, lr)

    # Set params
    with np.load(folder_name + 'generator_final.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(generator, param_values)

    with np.load(folder_name + 'discriminator_final.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(discriminator, param_values)

    with np.load(folder_name + 'encoder_z_final.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(encoder_z, param_values)

    with np.load(folder_name + 'encoder_y_final.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(encoder_y, param_values)


    num_people = 4 # use 4 people for each example figure (2 for swap)

    # Reconstruct + change attributes
    # Torch implementation gets running train batch norms std and mean, this uses fixed vals
    all_reconstructions = np.zeros((li*num_people, li*(lab_ln + 1), nc)).astype(np.float32)
    for index in range(0, num_people):
        image = load_files(X_files_test[index], 1, li)
        y_pred = encoder_y_test(image)
        z = encoder_z_test(image)

        # Duplicate Z 19 tiles
        z_permutations = np.zeros((lab_ln + 1,100)).astype(np.float32)
        for n in range(0, z_permutations.shape[0]):
            z_permutations[n,:] = z

        # Create y matrix
        y_permutations = modify_y(y_pred, False)

        # Generate images
        generated_ims = gen_fn(z_permutations, y_permutations)

        # Map reconstructions to main image
        for n in range(0, generated_ims.shape[0]):

            for chan in range(0, nc):
                all_reconstructions[li*index: li*index + li, li*n,: li*n + li, chan] = generated_ims[n, chan, :, : ]

    # Plot the reconstruction
    fig, ax = plt.subplots()

    ax.set_yticks(np.arange(0, li * num_people, li) + (li / 2), minor=False)
    ax.set_xticks(np.arange(0, li * len(labels) + li, li) + (li / 2), minor=False)
    ax.invert_yaxis()

    ax.set_xlabel('Description')
    ax.set_ylabel('Person')
    ax.set_title("Sample Generated Images")

    ax.set_xticklabels(labels, rotation='vertical', minor=False)
    plt.imshow(all_reconstructions)

    fig.savefig(folder_name + 'images/reconstructions.png')
    plt.close(fig)

    # Swap
    image_1 = load_files(X_files_test[0], 1, li)
    image_2 = load_files(X_files_test[1], 1, li)
    y_1 = encoder_y_test(image_1)
    z_1 = encoder_z_test(image_1)
    y_2 = encoder_y_test(image_2)
    z_2 = encoder_z_test(image_2)

    num_pairs = 1
    swap_image = np.zeros((li * num_pairs * 2, li * 3, nc)).astype(np.float32)
    for chan in range(0, nc):
        swap_image[0:li, 0:li, chan] = image_1[chan, :, :]
        swap_image[li:li + li, 0:li, chan] = image_2[chan, :, :]

    # Swaps for first image
    z_matrix = np.zeros((2, 100)).astype(np.float32)
    y_matrix = np.zeros((2, lab_ln)).astype(np.float32)
    for n in range(0, 2):
        z_matrix[n, :] = z_1

    y_matrix[0, :] = y_1
    y_matrix[1, :] = y_2

    ims = gen_fn(z_matrix, y_matrix)
    for n in range(1, 3):
        for chan in range(0, nc):
            swap_image[0:li, n*li:n*li + li, chan] = ims[n - 1, chan, :, :]

    # Swaps for second image
    z_matrix = np.zeros((2, 100)).astype(np.float32)
    y_matrix = np.zeros((2, lab_ln)).astype(np.float32)
    for n in range(0, 2):
        z_matrix[n, :] = z_2

    y_matrix[0, :] = y_2
    y_matrix[1, :] = y_1

    ims = gen_fn(z_matrix, y_matrix)
    for n in range(1, 3):
        for chan in range(0, nc):
            swap_image[li:li + li, n * li:n * li + li, chan] = ims[n - 1, chan, :, :]

    # Plot the swapped images
    x_lab = ['Original', 'Reconstruction', 'Swapped y']
    fig, ax = plt.subplots()

    ax.set_yticks(np.arange(0, li * num_pairs * 2, li) + (li / 2), minor=False)
    ax.set_xticks(np.arange(0, li * 3, li) + (li / 2), minor=False)
    ax.invert_yaxis()

    ax.set_ylabel('Person')
    ax.set_title("Swapped Images")

    ax.set_xticklabels(x_lab, rotation='vertical', minor=False)
    plt.imshow(all_reconstructions)

    fig.savefig(folder_name + 'images/swapped.png')
    plt.close(fig)

    # Interpolation
    image_1 = load_files(X_files_test[0], 1, li)
    image_2 = load_files(X_files_test[1], 1, li)
    y_1 = encoder_y_test(image_1)
    z_1 = encoder_z_test(image_1)
    y_2 = encoder_y_test(image_2)
    z_2 = encoder_z_test(image_2)

    # Interpolate y and z
    n_inter = 10
    y_inter = interpolate_vector(y_1, y_2, n_inter)
    z_inter = interpolate_vector(z_1, z_2, n_inter)
    interpolations = np.zeros((li, li* n_inter + li, nc)).astype(np.float32)

    # Generate interpolation images
    ims = gen_fn(z_inter, y_inter)

    for n in range(0, ims.shape[0]):
        for chan in range(0, nc):
            interpolations[:, li*n:li*n + li, chan] = ims[n, chan, :, :]

    # Plot interpolation
    fig, ax = plt.subplots()

    ax.set_yticks(np.arange(0, li * num_people, 1) + (li / 2), minor=False)
    ax.set_xticks(np.arange(0, li * n_inter + li, li) + (li / 2), minor=False)
    ax.invert_yaxis()

    ax.set_title("Interpolation between 2 people")

    plt.imshow(interpolations)

    fig.savefig(folder_name + 'images/interpolation.png')
    plt.close(fig)
