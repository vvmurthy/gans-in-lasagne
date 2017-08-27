# graphs_conditions - makes graphs for training methods that use y label data
import numpy as np
import matplotlib.pyplot as plt
from gen_utils import deprocess_image


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


# Show reconstructions from generator(encoder(im))
def show_reconstructions(images, reconstructions, li, nc, epoch, filename):

    num_examples = 10
    labels = ['Original', 'Reconstruction']
    image = np.zeros((li * num_examples, li * 2, nc)).astype(np.float32)

    for n in range(0, num_examples):

        select_im = deprocess_image(images[n, :, :, :], li, nc)
        select_rec = deprocess_image(reconstructions[n, :, :, :], li, nc)

        image[li*n: li*n + li, 0: li , :] = select_im
        image[li * n: li * n + li, li: 2*li, :] = select_rec

    fig, ax = plt.subplots()

    ax.set_xticks(np.arange(0, 64 * len(labels), 64) + (64 / 2), minor=False)
    ax.set_yticks([])

    ax.set_title("Example Generated Images: Epoch " + str(epoch))

    ax.set_xticklabels(labels, rotation='vertical', minor=False)
    
    if nc == 1:
        plt.imshow(np.squeeze(image), cmap='gray')
    else:
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

    # For black and white (1 channel) data, we must drop singleton dimension
    image = np.squeeze(image)

    ax.set_xlabel('Label Categories')
    ax.set_xticks(np.arange(0, li * len(labels), li) + (li / 2), minor=False)
    ax.set_yticks([])
    ax.invert_yaxis()

    ax.set_title("Example Generated Images: Epoch " + str(epoch))

    ax.set_xticklabels(labels, rotation='vertical', minor=False)
    if nc == 1:
        plt.imshow(np.squeeze(image), cmap='gray')
    else:
        plt.imshow(image)

    fig.savefig(filename)
    plt.close('all')


