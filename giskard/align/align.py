import numpy as np
from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
import time


def joint_procrustes(
    data1,
    data2,
    method="orthogonal",
    seeds=None,
    swap=False,
    verbose=False,
):
    n = len(data1[0])
    if method == "orthogonal":
        procruster = OrthogonalProcrustes()
    elif method == "transport":
        if seeds is None:
            procruster = SeedlessProcrustes(init="sign_flips")
        else:
            paired_inds1 = seeds[0]
            paired_inds2 = seeds[1]
            X1_paired = data1[0][paired_inds1, :]
            X2_paired = data2[0][paired_inds2, :]
            if swap:
                Y1_paired = data1[1][paired_inds2, :]
                Y2_paired = data2[1][paired_inds1, :]
            else:
                Y1_paired = data1[1][paired_inds1, :]
                Y2_paired = data2[1][paired_inds2, :]
            data1_paired = np.concatenate((X1_paired, Y1_paired), axis=0)
            data2_paired = np.concatenate((X2_paired, Y2_paired), axis=0)
            op = OrthogonalProcrustes()
            op.fit(data1_paired, data2_paired)
            procruster = SeedlessProcrustes(
                init="custom",
                initial_Q=op.Q_,
                optimal_transport_eps=1.0,
                optimal_transport_num_reps=100,
                iterative_num_reps=10,
            )
    data1 = np.concatenate(data1, axis=0)
    data2 = np.concatenate(data2, axis=0)
    currtime = time.time()
    data1_mapped = procruster.fit_transform(data1, data2)
    if verbose > 1:
        print(f"{time.time() - currtime:.3f} seconds elapsed for SeedlessProcrustes.")
    data1 = (data1_mapped[:n], data1_mapped[n:])
    return data1