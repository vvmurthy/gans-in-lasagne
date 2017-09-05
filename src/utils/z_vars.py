"""
z_vars = methods for creating z variable
(typically for generator input) 
"""
import numpy as np


def z_var_uniform(bz, num_hidden):
    """
    utils.z_var_uniform(bz, num_hidden)
    Creates Z vector from uniform distribution, values [-1, 1]

    Parameters
    ----------
    bz : int
        Batchsize of z vectors.
    num_hidden : int
        Length of z vector. Also called number of hidden units, or encoding state
        in some papers.
    Returns
    -------
    z : 2D :class:``NdArray``
        A matrix of z vectors of size ``[bz, num_hidden]``
    """
    z = np.array(np.random.uniform(-1, 1, (bz, num_hidden))).astype(np.float32)
    return z


