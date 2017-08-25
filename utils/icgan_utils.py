# icgan_utils - utilities only used for training icgan
import numpy as np


# Binarize y vector given certain threshold
def binarize_y_celeba(y, threshhold):

    type_hair = [0, 15, 16]
    color_hair = [2, 3, 4, 7]

    hair_type_index = type_hair[np.argmax(y[type_hair])]
    color_type_index = color_hair[np.argmax(y[color_hair])]
    
    y[type_hair] = -1
    y[hair_type_index] = 1
    
    y[color_hair] = -1
    y[color_type_index] = 1

    y[y <= threshhold] = -1
    y[y > threshhold] = 1
    return y

# Binarize y in a mutually exclusive manner
def binarize_y(y, threshhold):
    top_preds = np.argmax(y)
    y[:] = -1
    y[top_preds] = 1
    return y


# Generates 18 permutations of y
# y is a vector here
def modify_y_celeba(y, binarize):

    # Binarize y
    if binarize:
        y = binarize_y_celeba(y, 0)

    type_hair = [0, 15, 16]
    color_hair = [2, 3, 4, 7]
    y_vars = np.zeros((19,18)).astype(np.float32)

    y_vars[0, :] = y

    for n in range(0, y.shape[0]):
        temp = np.copy(y)

        if n in type_hair:
            temp[type_hair] = -1
            temp[n] = 1
        elif n in color_hair:
            temp[color_hair] = -1
            temp[n] = 1
        else:
            temp[n] = 1
        y_vars[n + 1,:] = temp

    return y_vars

# Input is a vector y, returns matrix of y variations
# Y is assumed to be mutually exclusive here
def modify_y(y, binarize):
    # Binarize y
    if binarize:
        y = binarize_y(y, 0)

    y_vars = np.zeros((y.shape[0] + 1, y.shape[0])).astype(np.float32)

    y_vars[0, :] = y

    indices = np.arange(0, y.shape[0])
    for n in range(0, y.shape[0]):
        temp = np.copy(y)
        temp[indices] = -1
        temp[n] = 1
        y_vars[n + 1, :] = temp

    return y_vars


# Generates fake y vectors for a given minibatch
def randomize_y(y):
    for n in range(0, y.shape[0]):
        index = np.random.randint(0, y.shape[1])
        y[n, :, 0] = modify_y(y[n, :, 0], False)[index, :]

    return y.astype(np.float32)


# Generates fake y vectors for a given minibatch
def randomize_y_celeba(y):
    for n in range(0, y.shape[0]):
        index = np.random.randint(0, y.shape[1])
        y[n, :, 0] = modify_y_celeba(y[n, :, 0], False)[index, :]

    return y.astype(np.float32)