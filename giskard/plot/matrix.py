import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import numpy as np


def dissimilarity_clustermap(
    dissimilarity,
    colors=None,
    palette=None,
    method="average",
    center=0,
    cmap="RdBu_r",
    invert=False,
    **kwargs
):
    if invert:
        cluster_dissimilarity = 1 - dissimilarity
    else:
        cluster_dissimilarity = dissimilarity
    # since it assumes a distance/dissimilarity is input, the metric kwarg doesnt matter
    Z = linkage(squareform(cluster_dissimilarity), method=method)

    if palette is not None and colors is not None:
        colors = np.array(np.vectorize(palette.get)(colors))

    out = sns.clustermap(
        dissimilarity,
        row_linkage=Z,
        col_linkage=Z,
        cmap=cmap,
        center=center,
        row_colors=colors,
        col_colors=colors,
        xticklabels=False,
        yticklabels=False,
        **kwargs,
    )
    return out
