import numpy as np
import matplotlib.pyplot as plt


# Batch iterator for conditional (with y label) data
def iterate_minibatches_conditional(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# Batch iterator for unconditional (without y label) data
# Also used to generate ids for batches in memory
def iterate_minibatches_unconditional(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]


# Also used to generate ids for batches in memory
def iterate_membatches(inputs, batchsize, dataset_loader, li, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield dataset_loader(inputs[excerpt], inputs[excerpt].shape[0], li)


# Calculates flattened dimension for convolutional layer to fully connected layer
def product(tuple_dims):
    prod = 1
    for dim in tuple_dims:
        prod = prod*dim
    return prod


def show_training_stats_graph(gen_train_err, dis_train_err_real, dis_train_err_fake, num_epochs, filename):
    # Make X labels
    labels = []
    for n in range(0, num_epochs):
        labels.append(str(n))

    # Plot the 3 different subplots
    x = np.arange(0, num_epochs)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.set_title("Training Stats for GAN")
    ax1.plot(x, gen_train_err)
    ax1.set_ylabel('Generator Train Error')

    ax2.plot(x, dis_train_err_real)
    ax2.set_ylabel('Discriminator Error on Real Images')

    ax3.plot(x, dis_train_err_fake)
    ax3.set_ylabel('Discriminator Error on Fake Images')

    ax3.set_xlabel('Epochs completed')

    fig.savefig(filename)
    plt.close(fig)


# Shows training stats for encoders
def show_encoder_stats_graph(encoder_z_loss, encoder_y_loss, num_epochs, filename):

    # Make X labels
    labels = []
    for n in range(0, num_epochs):
        labels.append(str(n))

    # Plot the 2 different subplots
    x = np.arange(0, num_epochs)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title("Training Stats for Encoders")
    ax1.plot(x, encoder_z_loss)
    ax1.set_ylabel('Encoder Z Loss')

    ax2.plot(x, encoder_y_loss)
    ax2.set_ylabel('Encoder Y Loss')

    ax2.set_xlabel('Epochs completed')

    fig.savefig(filename)
    plt.close(fig)


# Converts an image to [0, 1] range with n x n x c dimensions (for image saving)    
def deprocess_image(image, li, nc):
    image = (image / np.float32(2) + np.float32(0.5))
    im = np.zeros((li, li, nc)).astype(np.float32)
    for chan in range(0, nc):
        im[:, :, chan] = image[chan, :, :]
    return im


# Show reconstructions from generator(encoder(im))
def show_reconstructions(images, reconstructions, li, nc, epoch, filename):

    num_examples = 10
    labels = ['Original', 'Reconstruction']
    image = np.zeros((li * num_examples, li * 2, nc)).astype(np.float32)

    for n in range(0, num_examples):

        select_im = deprocess_image(images[n, :, :, :], li)
        select_rec = deprocess_image(reconstructions[n, :, :, :])

        image[li*n: li*n + li, 0: li , :] = select_im
        image[li * n: li * n + li, li: 2*li, :] = select_rec

    fig, ax = plt.subplots()

    ax.set_xticks(np.arange(0, 64 * len(labels), 64) + (64 / 2), minor=False)
    ax.set_yticks([])

    ax.set_title("Example Generated Images: Epoch " + str(epoch))

    ax.set_xticklabels(labels, rotation='vertical', minor=False)
    plt.imshow(image)

    fig.savefig(filename)
    plt.close('all')


# Show examples from each label category
def show_examples(images, y, labels, li, nc, epoch, filename):

    num_examples = 10
    image = np.zeros((li * num_examples, li * len(labels), nc)).astype(np.float32)

    # Get rows for each ground truth value
    for n in range(0, len(labels)):
        select_images = images[y[:,n] == 1]

        if select_images.shape[0] >= num_examples:
            for example in range(0, num_examples):
                select_im =  deprocess_image(select_images[example, :, :, :], li, nc)
                image[li * example : li* example + li, n * li : n * li + li, :] = select_im

    fig, ax = plt.subplots()

    ax.set_xlabel('Label Categories')
    ax.set_xticks(np.arange(0, li * len(labels), li) + (li / 2), minor=False)
    ax.set_yticks([])
    ax.invert_yaxis()

    ax.set_title("Example Generated Images: Epoch " + str(epoch))

    ax.set_xticklabels(labels, rotation='vertical', minor=False)
    plt.imshow(image)

    fig.savefig(filename)
    plt.close('all')


# Show 100 generated images as examples
def show_examples_unlabeled(images, y, labels, li, nc, epoch, filename):

    num_examples_row = 10
    image = np.zeros((li * num_examples_row, li * len(labels), nc)).astype(np.float32)

    example_count = 0
    for n in range(0, num_examples_row):
            for example in range(0, num_examples_row):
                select_im = deprocess_image(images[example_count, :, :, :], li, nc)
                image[li * example : li* example + li, n * li : n * li + li, :] = select_im
                example_count += 1

    fig, ax = plt.subplots()

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title("Example Generated Images: Epoch " + str(epoch))

    plt.imshow(image)

    fig.savefig(filename)
    plt.close('all')


# Binarize y vector given certain threshold
def binarize_y(y, threshhold):

    type_hair = [0, 15, 16]
    color_hair = [2, 3, 4, 7]

    index = np.argmax(y[type_hair], axis =1)
    y[:, type_hair] = -1
    y[:, type_hair[index]] = 1

    index = np.argmax(y[color_hair], axis=1)
    y[:, color_hair] = -1
    y[:, color_hair[index]] = 1

    y[y <= threshhold] = -1
    y[y > threshhold] = 1
    return y


# Generates 18 permutations of y
# y is a vector here
def modify_y(y, binarize):

    # Binarize y
    if binarize:
        y = binarize_y(y, 0)

    type_hair = [0, 15, 16]
    color_hair = [2, 3, 4, 7]
    y_vars = np.zeros((19,18)).astype(np.float32)

    y_vars[0, :] = y

    for n in range(1, y.shape[0]):
        temp = np.copy(y)

        if n in type_hair:
            temp[type_hair] = -1
            temp[n] = 1
        elif n in color_hair:
            temp[color_hair] = -1
            temp[n] = 1
        else:
            temp[n] = temp[n] * -1
        y_vars[n,:] = temp

    return y_vars


# Generates fake y vectors for a given minibatch
def randomize_y(y):
    for n in range(0, y.shape[0]):
        index = np.random.randint(0, y.shape[1])
        y[n, :, 0] = modify_y(y[n, :, 0], False)[index, :]
    return y


def interpolate_vector(vect_1, vect_2, n_interpolations):
    weights = np.arange(0, 1 + np.float32(1 / n_interpolations), np.float32(1 / n_interpolations))
    vect = np.zeros(weights.shape[0] + 1, vect_1.shape[1]).astype(np.float32)

    for n in range(0, weights.shape[0]):
        vect[n, :] = vect_1[n, :] * weights[n] + vect_2[n, :] * (1 - weights[n])

    return vect
