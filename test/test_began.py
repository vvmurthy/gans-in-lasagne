import os
import lasagne
import matplotlib.pyplot as plt
import numpy as np

from utils.gen_utils import interpolate_vector, deprocess_image
from utils.graphs_unconditional import show_examples_unlabeled

from models.build_began import make_train_fns


# Benchmark on test data
def test_began(configuration):
    
    # seed indices generation
    np.random.seed(115)

    # Check that models do exist
    assert(os.path.isfile(configuration['folder_name'] + '/models/generator_final.npz')), \
        ("Generator in " + configuration['folder_name'] + " Does not exist")
    assert (os.path.isfile(configuration['folder_name'] + '/models/discriminator_final.npz')), \
        ("Discriminator in " + configuration['folder_name'] + " Does not exist")

    # Set hyperparameters
    li = configuration['li']
    nc = configuration['nc']
    bz = configuration['bz']
    num_hidden = configuration['num_hidden']
    folder_name = configuration['folder_name']
    offset = configuration['offset']
    gamma = configuration['gamma']
    num_filters = configuration['num_filters']
    
    # Set dataset
    z_var = configuration['z_var']

    # Build GAN + Encoder
    generator, discriminator, gen_train_fn, gen_fn, dis_train_fn = make_train_fns(li, gamma, num_filters, num_hidden, offset)

    # Set params
    with np.load(folder_name + '/models/generator_final.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(generator['gen_out'], param_values)

    with np.load(folder_name + '/models/discriminator_final.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(discriminator['out'], param_values)
    
    # Generate several random images
    num_examples_row = 15        
    max_ims = int(num_examples_row ** 2 / bz) + 1
    val_ims = np.zeros((max_ims * bz, nc, li, li))
    for st in range(0, max_ims):
        noise = z_var(bz, num_hidden)
        val_ims[bz * st: bz * st + bz] = gen_fn(noise)

    show_examples_unlabeled(val_ims, num_examples_row, li, nc, None, 
                            folder_name + '/images/examples.png')

    # Interpolation
    n_inter = 10
    num_people = 10

    interpolations = np.zeros((li * num_people/2, li* n_inter + li, nc)).astype(np.float32)
    for n in range(0, num_people, 2):

        z_1 = z_var(1, num_hidden)[0]
        z_2 = z_var(1, num_hidden)[0]

        # Interpolate z
        z_inter = interpolate_vector(z_1, z_2, n_inter)
        
        # Generate interpolation images
        ims = gen_fn(z_inter)
        
        for q in range(0, ims.shape[0]):
                interpolations[li*(n/2): li*(n/2) + li, li*q:li*q + li, :] = deprocess_image(ims[q, :, :, :], li, nc)

    # Plot interpolation
    fig, ax = plt.subplots()

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_title("Interpolation between 2 Images")

    if nc == 1:
        plt.imshow(np.squeeze(interpolations), cmap='gray')
    else:
        plt.imshow(interpolations)

    fig.savefig(folder_name + '/images/interpolation.png', bbox_inches='tight')
    plt.close(fig)
