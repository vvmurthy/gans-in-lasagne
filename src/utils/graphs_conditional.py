# graphs_conditions - makes graphs for training methods that use y label data
import numpy as np
import matplotlib.pyplot as plt
from gen_utils import deprocess_image


def show_encoder_stats_graph(encoder_z_loss, encoder_y_loss, num_epochs, filename):
    """
    utils.show_encoder_stats_graph(encoder_z_loss, encoder_y_loss, num_epochs, filename)
    Interpolates between two vectors to create a matrix of interpolated vectors.

    Parameters
    ----------
    encoder_z_loss : 1D :class:``NdArray``
        Per Epoch losses of Encoder Z.
    encoder_y_loss : 1D :class:``NdArray``
        Per Epoch losses of Encoder y.
    num_epochs : int
        Number of epochs completed in training. Used to print graph with
        appropriate x scale.
    filename : string
        Absolute path filename for where to store the generated figure.
    """

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


# Show reconstructions from generator(encoder(im))
def show_reconstructions(images, reconstructions, li, nc, epoch, filename):
    """
    utils.show_reconstructions(images, reconstructions, li, nc, epoch, filename)
    Shows original images and their reconstructions side-by-side in a figure,
    then saves the figure.

    Parameters
    ----------
    images : 4D :class:``NdArray``
        Original images. Assumed to be preprocessed to format [-1, 1] range
        and ``batchsize x nc x li x li`` dimensions.
    reconstructions : 4D :class:``NdArray``
        Reconstructed images, typically from ``generator(encoder(image)).
        Assumed to be preprocessed to format [-1, 1] range
        and ``batchsize x nc x li x li`` dimensions.
    li : int
        Length of the image to be processed.
    nc : int
        Number of channels of image to be processed.
    epoch : int or ``None``
        Which epoch the reconstructions are from. This will display in the header
        of the figure. Set to ``None`` to not display epoch number in figure header.
    filename : string
        Absolute path filename for where to store the generated figure.
    """

    num_examples = 10
    labels = ['Original', 'Reconstruction']
    image = np.zeros((li * num_examples, li * 2, nc)).astype(np.float32)

    for n in range(0, num_examples):

        select_im = deprocess_image(images[n, :, :, :], li, nc)
        select_rec = deprocess_image(reconstructions[n, :, :, :], li, nc)

        image[li*n: li*n + li, 0: li , :] = select_im
        image[li * n: li * n + li, li: 2*li, :] = select_rec

    fig, ax = plt.subplots()

    ax.set_xticks(np.arange(0, li * len(labels), li) + (li / 2), minor=False)
    ax.set_yticks([])

    if epoch is not None:
        ax.set_title("Example Reconstructions: Epoch " + str(epoch))
    else:
        ax.set_title("Example Reconstructions")


    ax.set_xticklabels(labels, rotation='vertical', minor=False)
    
    if nc == 1:
        plt.imshow(np.squeeze(image), cmap='gray')
    else:
        plt.imshow(image)

    fig.savefig(filename)
    plt.close('all')


# Show examples from each label category
def show_examples(images, y, labels, li, nc, epoch, filename):
    """
    utils.show_reconstructions(images, y, labels, li, nc, epoch, filename)
    Shows positive examples from each label category then saves figure to
    specified file.

    Parameters
    ----------
    images : 4D :class:``NdArray``
        Original images. Assumed to be preprocessed to format [-1, 1] range
        and ``batchsize x nc x li x li`` dimensions.
    y : 2D :class:``NdArray``
        Binarized labels, where 1 is assumed to be the positive label.
    labels : :class:`List`
        Label descriptors. Order should follow order of labels in ``y``.
    li : int
        Length of the image to be processed.
    nc : int
        Number of channels of image to be processed.
    epoch : int or ``None``
        Which epoch the reconstructions are from. This will display in the header
        of the figure. Set to ``None`` to not display epoch number in figure header.
    filename : string
        Absolute path filename for where to store the generated figure.
    """

    num_examples = 10
    image = np.zeros((li * num_examples, li * len(labels), nc)).astype(np.float32)

    # Get rows for each ground truth value
    for n in range(0, len(labels)):
        select_images = images[y[:,n] == 1]

        if select_images.shape[0] >= num_examples:
            for example in range(0, num_examples):
                select_im = deprocess_image(select_images[example, :, :, :], li, nc)
                image[li * example : li* example + li, n * li : n * li + li, :] = select_im

    fig, ax = plt.subplots()

    # For black and white (1 channel) data, we must drop singleton dimension
    image = np.squeeze(image)

    ax.set_xlabel('Label Categories')
    ax.set_xticks(np.arange(0, li * len(labels), li) + (li / 2), minor=False)
    ax.set_yticks([])
    ax.invert_yaxis()

    if epoch is not None:
        ax.set_title("Example Generated Images: Epoch " + str(epoch))
    else:
        ax.set_title("Example Generated Images")

    ax.set_xticklabels(labels, rotation='vertical', minor=False)
    if nc == 1:
        plt.imshow(np.squeeze(image), cmap='gray')
    else:
        plt.imshow(image)

    fig.savefig(filename)
    plt.close('all')
