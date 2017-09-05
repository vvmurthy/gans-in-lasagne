import numpy as np


def iterate_minibatches_conditional(inputs, targets, batchsize, shuffle=False):
    """
    utils.iterate_minibatches_conditional(inputs, targets, batchsize, shuffle=False)

    Given a set of loaded images and their y labels, returns a subset of size ``batchsize``.

    Parameters
    ----------
    inputs : :class:``NdArray``
        Input images to the batch iterator.
    targets : :class:``NdArray``
        Corresponding targets to ``inputs``
    batchsize : int
        Number of images to yield in each iteration.
    shuffle : bool
        Whether to shuffle (return random images) or to return images in order. Typically
         shuffling is used in training stages, and disabled in testing. By
        default, is set to False.
    Returns
    -------
    inputs[excerpt] : :class:``NdArray``
        A subset of the parameter ``inputs`` of size ``batchsize``.
    targets[excerpt] : :class:``NdArray``
        The corresponding subset of labels for ``input[excerpt]``
    """
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


def iterate_minibatches_unconditional(inputs, batchsize, shuffle=False):
    """
    utils.iterate_minibatches_unconditional(inputs, batchsize, shuffle=False)

    Given a set of loaded images without y labels, returns a subset of size ``batchsize``.

    Parameters
    ----------
    inputs : :class:``NdArray``
        Input images to the batch iterator.
    batchsize : int
        Number of images to yield in each iteration.
    shuffle : bool
        Whether to shuffle (return random images) or to return images in order. Typically
         shuffling is used in training stages, and disabled in testing. By
        default, is set to False.
    Returns
    -------
    inputs[excerpt] : :class:``NdArray``
        A subset of the parameter ``inputs`` of size ``batchsize``.
    """
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]


def iterate_membatches(inputs, targets, batchsize, dataset_loader, shuffle=False):
    """
    utils.iterate_membatches(inputs, targets, batchsize, dataset_loader, shuffle=False)

    Given a set of absolute image paths and their y labels, returns a subset of loaded
    images and their corresponding filenames. Used when memory in computer is limited /
    dataset is large.

    Parameters
    ----------
    inputs : :class:``NdArray``
        Input filenames for batch iterator. Filenames are assumed to be absolute.
    targets : :class:``NdArray``
        Corresponding targets to ``inputs``
    batchsize : int
        Number of images to yield in each iteration.
    dataset_loader : Function
        Function (typically ``load_files``) used to load and preprocess the images.
        the ``dataset_loader`` function is specific to a given dataset.
    shuffle : bool
        Whether to shuffle (return random images) or to return images in order. Typically
         shuffling is used in training stages, and disabled in testing. By
        default, is set to False.
    Returns
    -------
    inputs[excerpt] : :class:``NdArray``
        A subset of the parameter ``inputs`` of size ``batchsize``, loaded in and preprocessed.
    targets[excerpt] : :class:``NdArray``
        The corresponding subset of labels for ``input[excerpt]``
    """
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield dataset_loader(inputs[excerpt], inputs[excerpt].shape[0]), targets[excerpt].astype(np.float32)


# Generates images to be loaded into memory
# This version uses unconditional data
def iterate_membatches_unconditional(inputs, batchsize, dataset_loader, shuffle=False):
    """
    utils.iterate_membatches_unconditional(inputs, batchsize, dataset_loader, shuffle=False)

    Given a set of absolute image paths without y labels, returns a subset of loaded
    images and their corresponding filenames. Used when memory in computer is limited /
    dataset is large.

    Parameters
    ----------
    inputs : :class:``NdArray``
        Input filenames for batch iterator. Filenames are assumed to be absolute.
    batchsize : int
        Number of images to yield in each iteration.
    dataset_loader : Function
        Function (typically ``load_files``) used to load and preprocess the images.
        the ``dataset_loader`` function is specific to a given dataset.
    shuffle : bool
        Whether to shuffle (return random images) or to return images in order. Typically
         shuffling is used in training stages, and disabled in testing. By
        default, is set to False.
    Returns
    -------
    inputs[excerpt] : :class:``NdArray``
        A subset of the parameter ``inputs`` of size ``batchsize``, loaded in and preprocessed.
    """
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield dataset_loader(inputs[excerpt], inputs[excerpt].shape[0])


