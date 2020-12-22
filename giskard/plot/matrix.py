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
    **kwargs
):
    # since it assumes a distance/dissimilarity is input, the metric kwarg doesnt matter
    Z = linkage(squareform(dissimilarity), method=method)

    if palette is not None and colors is not None:
        colors = np.vectorize(palette.get)(colors)

    out = sns.clustermap(
        dissimilarity,
        row_linkage=Z,
        col_linkage=Z,
        cmap=cmap,
        center=center,
        row_colors=colors,
        col_colors=colors,
        **kwargs
    )
    return out
