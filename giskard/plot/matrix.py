import colorcet as cc
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

simple_colors = {0: "black", 1: "lightgrey"}


def dissimilarity_clustermap(
    dissimilarity,
    colors=None,
    palette=None,
    method="ward",
    center=0,
    cmap="RdBu_r",
    invert=False,
    cut=False,
    fcluster_palette="simple",
    criterion="distance",
    t=0.5,
    cut_line_kws=None,
    squarelines=True,
    squareline_kws={},
    **kwargs,
):
    if invert:
        cluster_dissimilarity = 1 - dissimilarity
    else:
        cluster_dissimilarity = dissimilarity
    # since it assumes a distance/dissimilarity is input, the metric kwarg doesnt matter
    Z = linkage(squareform(cluster_dissimilarity), method=method)

    if palette is not None and colors is not None:
        colors = np.array(np.vectorize(palette.get)(colors))

    if cut:
        flat_labels = fcluster(Z, t, criterion=criterion)

        if fcluster_palette == "simple":
            fcluster_colors = np.vectorize(lambda x: simple_colors[x % 2])(flat_labels)
        elif fcluster_palette == "glasbey_light":
            fcluster_palette = dict(zip(np.unique(flat_labels), cc.glasbey_light))
        if isinstance(fcluster_palette, dict):
            fcluster_colors = np.array(np.vectorize(fcluster_palette.get)(flat_labels))

        colors = [fcluster_colors, colors]

    clustergrid = sns.clustermap(
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
    inds = clustergrid.dendrogram_row.reordered_ind

    if cut and squarelines:
        plot_squarelines(flat_labels[inds], clustergrid.ax_heatmap, **squareline_kws)

    if cut and criterion == "distance":
        if cut_line_kws is None:
            cut_line_kws = dict(linewidth=1.5, linestyle=":", color="dodgerblue")
        clustergrid.ax_col_dendrogram.axhline(t, **cut_line_kws)
        clustergrid.ax_row_dendrogram.axvline(t, **cut_line_kws)

    return clustergrid


def plot_squarelines(
    labels, ax, linewidth=2, linestyle="-", color="darkgrey", **kwargs
):
    inds = pd.Series(data=np.arange(len(labels)))
    first_ind_map = inds.groupby(by=labels, sort=False).first()
    last_ind_map = inds.groupby(by=labels, sort=False).last()
    lines = []
    for label in first_ind_map.keys():
        first = first_ind_map[label]
        last = last_ind_map[label] + 1
        line = [
            [first, first],
            [last, first],
            [last, last],
            [first, last],
            [first, first],
        ]
        lines.append(line)

    lc = LineCollection(
        lines, linewidth=linewidth, linestyle=linestyle, color=color, **kwargs
    )
    ax.add_collection(lc)
    return lc


def scattermap(data, ax=None, legend=False, sizes=(5, 10), hue=None, **kws):
    r"""
    Draw a matrix using points instead of a heatmap. Helpful for larger, sparse
    matrices.
    Parameters
    ----------
    data : np.narray, scipy.sparse.csr_matrix, ndim=2
        Matrix to plot
    ax: matplotlib axes object, optional
        Axes in which to draw the plot, by default None
    legend : bool, optional
        [description], by default False
    sizes : tuple, optional
        min and max of dot sizes, by default (5, 10)
    spines : bool, optional
        whether to keep the spines of the plot, by default False
    border : bool, optional
        [description], by default True
    **kws : dict, optional
        Additional plotting arguments
    Returns
    -------
    ax: matplotlib axes object, optional
        Axes in which to draw the plot, by default None
    """

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(20, 20))
    n_verts = data.shape[0]
    inds = np.nonzero(data)
    edges = np.squeeze(np.asarray(data[inds]))
    scatter_df = pd.DataFrame()
    scatter_df["weight"] = edges
    scatter_df["x"] = inds[1]
    scatter_df["y"] = inds[0]
    if hue is not None:
        scatter_df["hue"] = hue
    sns.scatterplot(
        data=scatter_df,
        x="x",
        y="y",
        size="weight",
        legend=legend,
        sizes=sizes,
        ax=ax,
        linewidth=0,
        hue=hue,
        **kws,
    )
    ax.set_xlim((-0.5, n_verts - 0.5))
    ax.set_ylim((n_verts - 0.5, -0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")
    return ax
