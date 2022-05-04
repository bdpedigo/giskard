import time
from functools import wraps

import numpy as np
from graspologic.types import Tuple
from numba import jit
from ot import sinkhorn
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state


def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


@parametrized
def write_status(f, msg, level):
    @wraps(f)
    def wrap(*args, **kw):
        obj = args[0]
        verbose = obj.verbose
        if level <= verbose:
            total_msg = (level - 1) * "   "
            total_msg += obj.status() + " " + msg
            print(total_msg)
        if (verbose >= 4) and (level <= verbose):
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            sec = te - ts
            output = total_msg + f" took {sec:.3f} seconds."
            print(output)
        else:
            result = f(*args, **kw)
        return result

    return wrap


class GraphMatchSolver(BaseEstimator):
    def __init__(
        self,
        A,
        B,
        AB=None,
        BA=None,
        similarity=None,
        partial_match=None,
        rng=None,
        init=1.0,
        verbose=False,  # 0 is nothing, 1 is loops, 2 is loops + sub, 3, is loops + sub + timing
        shuffle_input=True,
        maximize=True,
        maxiter=30,
        tol=0.01,
        transport=False,
        transport_regularizer=100,
        transport_tolerance=5e-2,
        transport_implementation="pot",
        transport_maxiter=500,
    ):
        # TODO more input checking
        self.rng = check_random_state(rng)
        self.init = init
        self.verbose = verbose
        self.shuffle_input = shuffle_input
        self.maximize = maximize
        self.maxiter = maxiter
        self.tol = tol

        self.transport = transport
        self.transport_regularizer = transport_regularizer
        self.transport_tolerance = transport_tolerance
        self.transport_implementation = transport_implementation
        self.transport_maxiter = transport_maxiter

        if maximize:
            self.obj_func_scalar = -1
        else:
            self.obj_func_scalar = 1

        if partial_match is None:
            partial_match = np.array([[], []]).astype(int).T
            self._seeded = False
        else:
            self._seeded = True
        self.partial_match = partial_match

        # TODO input validation
        # TODO seeds
        # A, B, partial_match = _common_input_validation(A, B, partial_match)

        # TODO similarity
        # if S is None:
        #     S = np.zeros((A.shape[0], B.shape[1]))
        # S = np.atleast_2d(S)

        # TODO padding

        # TODO make B always bigger

        # convert everything to make sure they are 3D arrays (first dim is layer)
        A = _check_input_matrix(A)
        B = _check_input_matrix(B)
        AB = _check_input_matrix(AB)
        BA = _check_input_matrix(BA)

        self.n_A = A[0].shape[0]
        self.n_B = B[0].shape[0]

        if AB is None:
            AB = np.zeros((A.shape[0], A.shape[1], B.shape[1]))
        if BA is None:
            BA = np.zeros((B.shape[0], B.shape[1], A.shape[1]))

        n_seeds = len(partial_match)
        self.n_seeds = n_seeds

        # set up so that seeds are first and we can grab subgraphs easily
        # TODO could also do this slightly more efficiently just w/ smart indexing?
        nonseed_A = np.setdiff1d(range(len(A[0])), partial_match[:, 0])
        nonseed_B = np.setdiff1d(range(len(B[0])), partial_match[:, 1])
        perm_A = np.concatenate([partial_match[:, 0], nonseed_A])
        perm_B = np.concatenate([partial_match[:, 1], nonseed_B])
        self._undo_perm_A = np.argsort(perm_A)
        self._undo_perm_B = np.argsort(perm_B)

        # permute each (sub)graph appropriately
        A = _permute_multilayer(A, perm_A, rows=True, columns=True)
        B = _permute_multilayer(B, perm_B, rows=True, columns=True)
        AB = _permute_multilayer(AB, perm_A, rows=True, columns=False)
        AB = _permute_multilayer(AB, perm_B, rows=False, columns=True)
        BA = _permute_multilayer(BA, perm_A, rows=False, columns=True)
        BA = _permute_multilayer(BA, perm_B, rows=True, columns=False)

        # split into subgraphs of seed-to-seed, seed-to-nonseed, etc.
        # main thing being permuted has no subscript
        self.A_ss, self.A_sn, self.A_ns, self.A = _split_matrix(A, n_seeds)
        self.B_ss, self.B_sn, self.B_ns, self.B = _split_matrix(B, n_seeds)
        self.AB_ss, self.AB_sn, self.AB_ns, self.AB = _split_matrix(AB, n_seeds)
        self.BA_ss, self.BA_sn, self.BA_ns, self.BA = _split_matrix(BA, n_seeds)

        self.n_unseed = self.B[0].shape[0]

        if similarity is None:
            similarity = np.zeros((self.n_A, self.n_B))

        similarity = similarity[perm_A][:, perm_B]
        self.S_ss, self.S_sn, self.S_ns, self.S = _split_matrix(similarity, n_seeds)

    def solve(self):
        self.n_iter = 0
        self.check_outlier_cases()

        P = self.initialize()
        self.compute_constant_terms()
        for n_iter in range(self.maxiter):
            self.n_iter = n_iter + 1

            gradient = self.compute_gradient(P)
            Q = self.compute_step_direction(gradient)
            alpha = self.compute_step_size(P, Q)

            # take a step in this direction
            P_new = alpha * P + (1 - alpha) * Q

            if self.check_converged(P, P_new):
                self.converged = True
                P = P_new
                break
            P = P_new

        self.finalize(P)

    # TODO
    def check_outlier_cases(self):
        pass

    @write_status("Initializing", 1)
    def initialize(self):
        if isinstance(self.init, float):
            n_unseed = self.n_unseed
            rng = self.rng
            J = np.ones((n_unseed, n_unseed)) / n_unseed
            # DO linear combo from barycenter
            K = rng.uniform(size=(n_unseed, n_unseed))
            # Sinkhorn balancing
            K = _doubly_stochastic(K)
            P = J * self.init + K * (1 - self.init)  # TODO check how defined in paper
        elif isinstance(self.init, np.ndarray):
            raise NotImplementedError()
            # TODO fix below
            # P0 = np.atleast_2d(P0)
            # _check_init_input(P0, n_unseed)
            # invert_inds = np.argsort(nonseed_B)
            # perm_nonseed_B = np.argsort(invert_inds)
            # P = P0[:, perm_nonseed_B]

        self.converged = False
        return P

    @write_status("Computing constant terms", 2)
    def compute_constant_terms(self):
        self.constant_sum = np.zeros(self.B.shape)
        if self._seeded:
            n_layers = len(self.A)
            ipsi = []
            contra = []
            for i in range(n_layers):
                ipsi.append(
                    self.A_ns[i] @ self.B_ns[i].T + self.A_sn[i].T @ self.B_sn[i]
                )
                contra.append(
                    self.AB_ns[i] @ self.BA_ns[i].T + self.BA_sn[i].T @ self.AB_sn[i]
                )
            ipsi = np.array(ipsi)
            contra = np.array(contra)
            self.ipsi_constant_sum = ipsi
            self.contra_constant_sum = contra
            self.constant_sum = ipsi + contra
        self.constant_sum += self.S

    @write_status("Computing gradient", 2)
    def compute_gradient(self, P):
        gradient = _compute_gradient(
            P, self.A, self.B, self.AB, self.BA, self.constant_sum
        )
        return gradient

    @write_status("Solving assignment problem", 2)
    def compute_step_direction(self, gradient):
        # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
        if self.transport:
            Q = self.linear_sum_transport(gradient)
        else:
            permutation = self.linear_sum_assignment(gradient)
            Q = np.eye(self.n_unseed)[permutation]
        return Q

    def linear_sum_assignment(self, P):
        row_perm = np.random.permutation(P.shape[1])
        undo_row_perm = np.argsort(row_perm)
        P_perm = P[row_perm]
        _, permutation = linear_sum_assignment(P_perm, maximize=self.maximize)
        return permutation[undo_row_perm]

    def linear_sum_transport(
        self,
        P,
    ):
        maximize = self.maximize
        reg = self.transport_regularizer

        power = -1 if maximize else 1
        lamb = reg / np.max(np.abs(P))
        if self.transport_implementation == "pot":
            ones = np.ones(P.shape[0])
            P_eps = sinkhorn(
                ones,
                ones,
                P,
                power / lamb,
                stopInnerThr=self.transport_tolerance,
                numItermax=self.transport_maxiter,
            )
        elif self.transport_implementation == "ds":
            P = np.exp(lamb * power * P)
            P_eps = _doubly_stochastic(
                P, self.transport_tolerance, self.transport_maxiter
            )
        return P_eps

    @write_status("Computing step size", 2)
    def compute_step_size(self, P, Q):
        a, b = _compute_coefficients(
            P,
            Q,
            self.A,
            self.B,
            self.AB,
            self.BA,
            self.A_ns,
            self.A_sn,
            self.B_ns,
            self.B_sn,
            self.AB_ns,
            self.AB_sn,
            self.BA_ns,
            self.BA_sn,
            self.S,
        )
        if a * self.obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = np.argmin([0, (b + a) * self.obj_func_scalar])
        return alpha

    def check_converged(self, P, P_new):
        return np.linalg.norm(P - P_new) / np.sqrt(self.n_unseed) < self.tol

    @write_status("Finalizing assignment", 1)
    def finalize(self, P):
        self.P_final_ = P

        permutation = self.linear_sum_assignment(P)
        permutation += len(self.partial_match[:, 1])  # TODO this is not robust
        permutation = np.concatenate((self.partial_match[:, 1], permutation))
        self.permutation_ = permutation

        score = self.compute_score(permutation)
        self.score_ = score

    def compute_score(*args):
        return 0

    def status(self):
        if self.n_iter > 0:
            return f"[Iteration: {self.n_iter}]"
        else:
            return "[Pre-loop]"


def _permute_multilayer(adjacency, permutation, rows=True, columns=True):
    for layer_index in range(len(adjacency)):
        layer = adjacency[layer_index]
        if rows:
            layer = layer[permutation]
        if columns:
            layer = layer[:, permutation]
        adjacency[layer_index] = layer
    return adjacency


def _check_input_matrix(A):
    if isinstance(A, np.ndarray) and (np.ndim(A) == 2):
        A = np.expand_dims(A, axis=0)
        A = A.astype(float)
    if isinstance(A, list):
        A = np.array(A, dtype=float)
    return A


@jit(nopython=True)
def _compute_gradient(P, A, B, AB, BA, const_sum):
    n_layers = A.shape[0]
    grad = np.zeros_like(P)
    for i in range(n_layers):
        grad += (
            A[i] @ P @ B[i].T
            + A[i].T @ P @ B[i]
            + AB[i] @ P.T @ BA[i].T
            + BA[i].T @ P.T @ AB[i]
            + const_sum[i]
        )
    return grad


#


@jit(nopython=True)
def _compute_coefficients(
    P, Q, A, B, AB, BA, A_ns, A_sn, B_ns, B_sn, AB_ns, AB_sn, BA_ns, BA_sn, S
):
    R = P - Q
    # TODO make these "smart" traces like in the scipy code, couldn't hurt
    # though I don't know how much Numba cares

    n_layers = A.shape[0]
    a_cross = 0
    b_cross = 0
    a_intra = 0
    b_intra = 0
    for i in range(n_layers):
        a_cross += np.trace(AB[i].T @ R @ BA[i] @ R)
        b_cross += np.trace(AB[i].T @ R @ BA[i] @ Q) + np.trace(AB[i].T @ Q @ BA[i] @ R)
        b_cross += np.trace(AB_ns[i].T @ R @ BA_ns[i]) + np.trace(
            AB_sn[i].T @ BA_sn[i] @ R
        )
        a_intra += np.trace(A[i] @ R @ B[i].T @ R.T)
        b_intra += np.trace(A[i] @ Q @ B[i].T @ R.T) + np.trace(A[i] @ R @ B[i].T @ Q.T)
        b_intra += np.trace(A_ns[i].T @ R @ B_ns[i]) + np.trace(A_sn[i] @ R @ B_sn[i].T)

    a = a_cross + a_intra
    b = b_cross + b_intra
    b += np.sum(S * R)  # equivalent to S.T @ R

    return a, b


# REF: https://github.com/microsoft/graspologic/blob/dev/graspologic/match/qap.py
def _doubly_stochastic(P: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    # Adapted from @btaba implementation
    # https://github.com/btaba/sinkhorn_knopp
    # of Sinkhorn-Knopp algorithm
    # https://projecteuclid.org/euclid.pjm/1102992505

    max_iter = 1000
    c = 1 / P.sum(axis=0)
    r = 1 / (P @ c)
    P_eps = P

    for it in range(max_iter):
        if (np.abs(P_eps.sum(axis=1) - 1) < tol).all() and (
            np.abs(P_eps.sum(axis=0) - 1) < tol
        ).all():
            # All column/row sums ~= 1 within threshold
            break

        c = 1 / (r @ P)
        r = 1 / (P @ c)
        P_eps = r[:, None] * P * c

    return P_eps


def _split_matrix(
    matrices: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # definitions according to Seeded Graph Matching [2].
    if matrices.ndim == 2:
        matrices = [matrices]
    n_layers = len(matrices)
    seed_to_seed = []
    seed_to_nonseed = []
    nonseed_to_seed = []
    nonseed_to_nonseed = []
    for i in range(n_layers):
        X = matrices[i]
        upper, lower = X[:n], X[n:]
        seed_to_seed.append(upper[:, :n])
        seed_to_nonseed.append(upper[:, n:])
        nonseed_to_seed.append(lower[:, :n])
        nonseed_to_nonseed.append(lower[:, n:])
    seed_to_seed = np.array(seed_to_seed)
    seed_to_nonseed = np.array(seed_to_nonseed)
    nonseed_to_seed = np.array(nonseed_to_seed)
    nonseed_to_nonseed = np.array(nonseed_to_nonseed)
    return seed_to_seed, seed_to_nonseed, nonseed_to_seed, nonseed_to_nonseed
