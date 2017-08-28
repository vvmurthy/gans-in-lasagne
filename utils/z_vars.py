"""
z_vars = methods for creating z variable
(typically for generator input) 
"""
import numpy as np


def z_var_uniform(bz, num_hidden):
    return np.array(np.random.uniform(-1, 1, (bz, num_hidden))).astype(np.float32)


