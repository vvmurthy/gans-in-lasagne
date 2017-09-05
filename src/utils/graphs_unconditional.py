# graphs_unconditional - makes graphs for training methods that do not use y label data
# some functions may be used by conditional models as well
import numpy as np
import matplotlib.pyplot as plt
from gen_utils import deprocess_image


# Shows graph of training statistics
# First is always generator error, then discriminator error, then a third error
def show_training_stats_graph(gen_train_err, dis_train_err, third_error, num_epochs, filename,
                              second_error_title='Discriminator Error on Real Images',
                              third_error_title='Discriminator Error on Fake Images'):
    """
    utils.show_training_stats_graph(gen_train_err, dis_train_err, third_error, num_epochs, filename,
                              second_error_title='Discriminator Error on Real Images',
                              third_error_title='Discriminator Error on Fake Images')

    Shows a graph of error by each epoch, then saves figure to specified filename.

    Parameters
    ----------
    gen_train_err : 1D :class:``NdArray``
        Per epoch error of the generator.
    dis_train_err : 1D :class:``NdArray``
        Per epoch error of the discriminator.
    third_error : 1D :class:``NdArray``
        Optional third error. Some GANs split error of discriminator on fake / real images-
        this parameter can be used to separate error in the graphs.
    num_epochs : int
        number of epochs network was trained for. Used to generate X axis labels.
    filename : string
        Absolute path filename for where to store the generated figure.
    second_error_title : string
        Title of the ``dis_train_err`` graph. By default, is set to ``'Discriminator
        Error on Real Images'`` , which is the title used in IcGAN training.
    third_error_title : string
        Title of the ``third_error`` graph. By default, is set to ``'Discriminator
        Error on Fake Images'`` , which is the title used in IcGAN training.
    """
    # Make X labels
    labels = []
    for n in range(0, num_epochs):
        labels.append(str(n))

    # Plot the 3 different subplots
    x = np.arange(0, num_epochs)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.set_title("Training Stats for GAN")
    ax1.plot(x, gen_train_err)
    ax1.set_ylabel('Generator Error')

    ax2.plot(x, dis_train_err)
    ax2.set_ylabel(second_error_title)

    ax3.plot(x, third_error)
    ax3.set_ylabel(third_error_title)

    ax3.set_xlabel('Epochs completed')

    fig.savefig(filename)
    plt.close(fig)


def show_examples_unlabeled(images, num_examples_row, li, nc, epoch, filename):
    """
    utils.show_examples_unlabeled(images, num_examples_row, li, nc, epoch, filename)

    Shows sample of generated images, then saves figure to specified filename.

    Parameters
    ----------
    images : 4D :class:``NdArray``
        Generated images. Assumed to be in [-1, 1] range
        and ``batchsize x nc x li x li`` dimensions.
    num_examples_row : int
        number of examples to show in each row of the figure. The figure is square
        so the total number of generated examples shown will be ``num_examples_row ** 2``
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

    image = np.zeros((li * num_examples_row, li * num_examples_row, nc)).astype(np.float32)

    example_count = 0
    for n in range(0, num_examples_row):
            for example in range(0, num_examples_row):
                select_im = deprocess_image(images[example_count, :, :, :], li, nc)
                image[li * example : li* example + li, n * li : n * li + li, :] = select_im
                example_count += 1

    fig, ax = plt.subplots()

    ax.set_xticks([])
    ax.set_yticks([])

    if epoch is not None:
        ax.set_title("Example Generated Images: Epoch " + str(epoch))
    else:
        ax.set_title("Example Generated Images")

    if nc == 1:
        plt.imshow(np.squeeze(image), cmap='gray')
    else:
        plt.imshow(image)

    fig.savefig(filename, bbox_inches='tight')
    plt.close('all')
