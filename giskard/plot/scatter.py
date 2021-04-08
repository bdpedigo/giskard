import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP
from graspologic.embed import select_dimension
from .utils import soft_axis_off
import warnings


def simple_scatterplot(
    X,
    labels=None,
    palette="deep",
    ax=None,
    title="",
    legend=False,
    figsize=(10, 10),
    s=15,
    alpha=0.7,
    linewidth=0,
    spines_off=True,
    **kwargs
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    plot_df = pd.DataFrame(data=X[:, :2], columns=["0", "1"])
    plot_df["labels"] = labels
    sns.scatterplot(
        data=plot_df,
        x="0",
        y="1",
        hue="labels",
        palette=palette,
        ax=ax,
        s=s,
        alpha=alpha,
        linewidth=linewidth,
        **kwargs,
    )
    ax.set(title=title)
    if spines_off:
        soft_axis_off(ax, top=False, bottom=False, right=False, left=False)
    else:
        soft_axis_off(ax, top=False, bottom=True, right=False, left=True)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    if legend:
        # convenient default that I often use, places in the top right outside of plot
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    return ax


def simple_umap_scatterplot(
    X,
    labels=None,
    min_dist=0.75,
    n_neighbors=20,
    metric="euclidean",
    umap_kws={},
    palette="deep",
    ax=None,
    title="",
    legend=False,
    figsize=(10, 10),
    s=15,
    alpha=0.7,
    linewidth=0,
    scatter_kws={},
):
    umapper = UMAP(
        min_dist=min_dist, n_neighbors=n_neighbors, metric=metric, **umap_kws
    )
    warnings.filterwarnings("ignore", category=UserWarning, module="umap")
    umap_embedding = umapper.fit_transform(X)
    ax = simple_scatterplot(
        umap_embedding,
        labels=labels,
        palette=palette,
        ax=ax,
        title=r"UMAP $\circ$ " + title,
        legend=legend,
        figsize=figsize,
        s=s,
        alpha=alpha,
        linewidth=linewidth,
        **scatter_kws,
    )
    return ax


def textplot(x, y, text, ax=None, x_pad=0, y_pad=0, **kwargs):
    """Plot a iterables of x, y, text with matplotlib's ax.text"""
    if ax is None:
        ax = plt.gca()
    for x_loc, y_loc, s in zip(x, y, text):
        ax.text(
            x_loc + x_pad,
            y_loc + y_pad,
            s,
            transform=ax.transData,
            **kwargs,
        )


def screeplot(
    singular_values, check_n_components=None, ax=None, title="Screeplot", n_elbows=4
):
    if ax is None:
        ax = plt.gca()

    elbows, elbow_vals = select_dimension(
        singular_values[:check_n_components], n_elbows=n_elbows
    )

    index = np.arange(1, len(singular_values) + 1)

    sns.lineplot(x=index, y=singular_values, ax=ax, zorder=1)
    sns.scatterplot(
        x=elbows,
        y=elbow_vals,
        color="darkred",
        marker="x",
        ax=ax,
        zorder=2,
        s=80,
        linewidth=2,
    )
    textplot(
        elbows,
        elbow_vals,
        elbows,
        ax=ax,
        color="darkred",
        fontsize="small",
        x_pad=0.5,
        y_pad=0,
        zorder=3,
    )
    ax.set(title=title, xlabel="Index", ylabel="Singular value")
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    return ax
