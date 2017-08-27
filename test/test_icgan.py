import os

import lasagne
import matplotlib.pyplot as plt
import numpy as np

from utils.gen_utils import interpolate_vector, deprocess_image
from models.build_encoders import build_encoder_z, build_encoder_y
from models.build_gans import make_train_fns


# Benchmark on test data
def test_icgan(configuration):
    
    # seed indices generation
    np.random.seed(129)

    # Check that models do exist
    assert(os.path.isfile(configuration['folder_name'] + '/models/generator_final.npz')), \
        ("Generator in " + configuration['folder_name'] + " Does not exist")
    assert (os.path.isfile(configuration['folder_name'] + '/models/discriminator_final.npz')), \
        ("Discriminator in " + configuration['folder_name'] + " Does not exist")
    assert (os.path.isfile(configuration['folder_name'] + '/models/encoder_z_final.npz')), \
        ("Encoder Z in " + configuration['folder_name'] + " Does not exist")
    assert (os.path.isfile(configuration['folder_name'] + '/models/encoder_y_final.npz')), \
        ("Encoder y in " + configuration['folder_name'] + " Does not exist")

    # Set hyperparameters
    li = configuration['li']
    nc = configuration['nc']
    lr = configuration['lr']
    bz = configuration['bz']
    lab_ln = configuration['lab_ln']
    folder_name = configuration['folder_name']
    modify_y = configuration['modify_y']
    dataset_loader = configuration['dataset_loader']

    # Set dataset
    X_files_test = configuration['X_files_test']
    labels = configuration['labels']

    # Build GAN + Encoder
    generator, discriminator, gen_train_fn, gen_fn, dis_train_fn = make_train_fns(bz, li, nc, lab_ln)
    encoder_z, encoder_z_train, encoder_z_test = build_encoder_z(li, nc, lr)
    encoder_y, encoder_y_train, encoder_y_test = build_encoder_y(li, nc, lab_ln, lr)

    # Set params
    with np.load(folder_name + '/models/generator_final.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(generator['gen_out'], param_values)

    with np.load(folder_name + '/models/discriminator_final.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(discriminator['out'], param_values)

    with np.load(folder_name + '/models/encoder_z_final.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(encoder_z['out'], param_values)

    with np.load(folder_name + '/models/encoder_y_final.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(encoder_y['out'], param_values)


    num_people = 10 # use 4 people for reconstruction

    # Reconstruct + change attributes
    # Torch implementation gets running train batch norms std and mean, this uses fixed vals
    all_reconstructions = np.zeros((li*num_people, li*(lab_ln + 1), nc)).astype(np.float32)
    indices = np.random.randint(0, X_files_test.shape[0], num_people)
    for index in range(0, num_people):
        image = dataset_loader(X_files_test[indices[index]], 1, li)
        y_pred = np.squeeze(encoder_y_test(image)[0])
        z = np.squeeze(encoder_z_test(image)[0])

        # Duplicate Z
        z_permutations = np.zeros((lab_ln + 1,100)).astype(np.float32)
        for n in range(0, z_permutations.shape[0]):
            z_permutations[n,:] = z

        # Create y matrix
        y_permutations = np.expand_dims(modify_y(y_pred, True), axis=2)
            

        # Generate images
        generated_ims = gen_fn(z_permutations, y_permutations)

        # Map reconstructions to main image
        for n in range(0, generated_ims.shape[0]):
            all_reconstructions[li*index: li*index + li, li*n: li*n + li, :] = deprocess_image(generated_ims[n, :, :, : ], li, nc)

    # Plot the reconstruction
    fig, ax = plt.subplots()

    ax.set_yticks([])
    ax.set_xticks(np.arange(0, li * len(labels) + li, li) + (li / 2), minor=False)

    ax.set_xlabel('Description')
    ax.set_ylabel('Person')
    ax.set_title("Sample Generated Images")

    ax.set_xticklabels(['Reconstruction'] + labels, rotation='vertical', minor=False)
    ax.set_yticklabels([])
    
    if nc == 1:
        plt.imshow(np.squeeze(all_reconstructions), cmap='gray')
    else:
        plt.imshow(all_reconstructions)

    fig.savefig(folder_name + '/images/reconstructions.png')
    plt.close(fig)

    # Swap
    indices = np.random.randint(0, X_files_test.shape[0], 2)
    image_1 = dataset_loader(X_files_test[indices[0]], 1, li)
    image_2 = dataset_loader(X_files_test[indices[1]], 1, li)
    y_1 = np.squeeze(encoder_y_test(image_1)[0])
    z_1 = np.squeeze(encoder_z_test(image_1)[0])
    y_2 = np.squeeze(encoder_y_test(image_2)[0])
    z_2 = np.squeeze(encoder_z_test(image_2)[0])

    swap_image = np.zeros((li * 2, li * 3, nc)).astype(np.float32)
    swap_image[0:li, 0:li, :] = deprocess_image(image_1[0, :, :, :], li, nc)
    swap_image[li:li + li, 0:li, :] = deprocess_image(image_2[0, :, :, :], li, nc)

    # Swaps for first image
    z_matrix = np.zeros((2, 100)).astype(np.float32)
    y_matrix = np.zeros((2, lab_ln)).astype(np.float32)
    for n in range(0, 2):
        z_matrix[n, :] = z_1

    y_matrix[0, :] = y_1
    y_matrix[1, :] = y_2

    ims = gen_fn(z_matrix, np.expand_dims(y_matrix, axis=2))
    for n in range(1, 3):
        swap_image[0:li, n*li:n*li + li, :] = deprocess_image(ims[n - 1, :, :, :], li, nc)

    # Swaps for second image
    z_matrix = np.zeros((2, 100)).astype(np.float32)
    y_matrix = np.zeros((2, lab_ln)).astype(np.float32)
    for n in range(0, 2):
        z_matrix[n, :] = z_2

    y_matrix[0, :] = y_2
    y_matrix[1, :] = y_1

    ims = gen_fn(z_matrix, np.expand_dims(y_matrix, axis=2))
    for n in range(1, 3):
        swap_image[li:li + li, n * li:n * li + li, :] = deprocess_image(ims[n - 1, :, :, :], li, nc)

    # Plot the swapped images
    x_lab = ['Original', 'Reconstruction', 'Swapped y']
    fig, ax = plt.subplots()

    ax.set_yticks([])
    ax.set_xticks(np.arange(0, li * 3, li) + (li / 2), minor=False)
    ax.invert_yaxis()

    ax.set_title("Swapped Images")

    ax.set_xticklabels(x_lab, rotation='vertical', minor=False)
    ax.set_yticklabels([])
    
    if nc == 1:
        plt.imshow(np.squeeze(swap_image), cmap='gray')
    else:
        plt.imshow(swap_image)

    fig.savefig(folder_name + '/images/swapped.png')
    plt.close(fig)

    # Interpolation
    n_inter = 10
    indices = np.random.randint(0, X_files_test.shape[0], num_people)
    interpolations = np.zeros((li * num_people/2, li* n_inter + li, nc)).astype(np.float32)
    for n in range(0, num_people, 2):
    
        image_1 = dataset_loader(X_files_test[indices[n]], 1, li)
        image_2 = dataset_loader(X_files_test[indices[n + 1]], 1, li)
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
                interpolations[li*(n/2): li*(n/2) + li, li*q:li*q + li, :] = deprocess_image(ims[q, :, :, :], li, nc)

    # Plot interpolation
    fig, ax = plt.subplots()

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_title("Interpolation between 2 people")

    if nc == 1:
        plt.imshow(np.squeeze(interpolations), cmap='gray')
    else:
        plt.imshow(interpolations)

    fig.savefig(folder_name + '/images/interpolation.png')
    plt.close(fig)
