from .solver import GraphMatchSolver
from sklearn.utils import check_random_state
from joblib import Parallel, delayed


def graph_match(
    A,
    B,
    AB=None,
    BA=None,
    similarity=None,
    partial_match=None,
    random_state=None,
    n_init=1,
    init=1.0,
    verbose=False,
    shuffle_input=True,
    maximize=True,
    maxiter=30,
    tol=0.01,
    optimal_transport=False,
):
    pass

    # solver = GraphMatchSolver(
    #     A,
    #     B,
    #     AB=None,
    #     BA=None,
    #     similarity=None,
    #     partial_match=None,
    #     rng=None,
    #     init=1.0,
    #     verbose=False,
    #     shuffle_input=True,
    #     maximize=True,
    #     maxiter=30,
    #     tol=0.01,
    #     optimal_transport=False,
    # )

    # rng = check_random_state(self.random_state)
    # results = Parallel(n_jobs=self.n_jobs)(
    #     delayed(quadratic_assignment)(A, B, options={**options, **{"rng": r}})
    #     for r in rng.randint(np.iinfo(np.int32).max, size=self.n_init)
    # )
    # func = max if self.gmp else min
    # res = func(
    #     results,
    #     key=lambda x: x.fun,
    # )
