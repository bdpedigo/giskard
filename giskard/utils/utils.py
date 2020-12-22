import numpy as np


def careys_rule(X):
    """Get the number of singular values to check"""
    return int(np.ceil(np.log2(np.min(X.shape))))
