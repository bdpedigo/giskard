import numpy as np
from itertools import chain, combinations
from functools import wraps
import time


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


def timer(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        sec = te - ts
        output = f"Function {f.__name__} took {sec:.3f} seconds."
        print(output)
        return result

    return wrap


def get_random_seed(random_state):
    seed = random_state.integers(np.iinfo(np.int32).max)
    return seed
