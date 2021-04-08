import numpy as np
from itertools import chain, combinations


def careys_rule(X):
    """Get the number of singular values to check"""
    return int(np.ceil(np.log2(np.min(X.shape))))


def powerset(iterable, ignore_empty=True):
    # REF: https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(
        chain.from_iterable(combinations(s, r) for r in range(ignore_empty, len(s) + 1))
    )