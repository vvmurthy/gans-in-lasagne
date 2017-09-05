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


# Show 100 generated images as examples
def show_examples_unlabeled(images, num_examples_row, li, nc, epoch, filename):

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
