import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP

from graspologic.embed import select_dimension

from .utils import legend_upper_right, soft_axis_off


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
    **kwargs,
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
    singular_values,
    check_n_components=None,
    ax=None,
    title="Screeplot",
    n_elbows=4,
    label_elbows=True,
    label=None,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    elbows, elbow_vals = select_dimension(
        singular_values[:check_n_components], n_elbows=n_elbows
    )

    index = np.arange(1, len(singular_values) + 1)

    sns.lineplot(x=index, y=singular_values, ax=ax, zorder=1, label=label, **kwargs)
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
    if label_elbows:
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


def matched_stripplot(
    data,
    x=None,
    y=None,
    jitter=0.2,
    hue=None,
    match=None,
    ax=None,
    matchline_kws=None,
    order=None,
    **kwargs,
):
    data = data.copy()
    if ax is None:
        ax = plt.gca()

    if order is None:
        unique_x_var = data[x].unique()
    else:
        unique_x_var = order
    ind_map = dict(zip(unique_x_var, range(len(unique_x_var))))
    data["x"] = data[x].map(ind_map)
    if match is not None:
        groups = data.groupby(match)
        for _, group in groups:
            perturb = np.random.uniform(-jitter, jitter)
            data.loc[group.index, "x"] += perturb
    else:
        data["x"] += np.random.uniform(-jitter, jitter, len(data))

    sns.scatterplot(data=data, x="x", y=y, hue=hue, ax=ax, zorder=1, **kwargs)

    if match is not None:
        unique_match_var = data[match].unique()
        fake_palette = dict(zip(unique_match_var, len(unique_match_var) * ["black"]))
        if matchline_kws is None:
            matchline_kws = dict(alpha=0.2, linewidth=1)
        sns.lineplot(
            data=data,
            x="x",
            y=y,
            hue=match,
            ax=ax,
            legend=False,
            palette=fake_palette,
            zorder=-1,
            **matchline_kws,
        )
    ax.set(xlabel=x, xticks=np.arange(len(unique_x_var)), xticklabels=unique_x_var)
    return ax


def scatterplot(
    data,
    x=None,
    y=None,
    hue=None,
    shift=None,
    shift_bounds=(-0.1, 0.1),
    ax=None,
    shade=False,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
    data = data.copy()
    data["x_shift"] = data[x]
    if shift is not None:
        groups = data.groupby(shift, sort=False)
        shifts = np.linspace(shift_bounds[0], shift_bounds[1], len(groups))
        shifts = dict(zip(groups.groups.keys(), shifts))
        for group_key, group_data in groups:
            data.loc[group_data.index, "x_shift"] += shifts[group_key]
    sns.scatterplot(data=data, x="x_shift", y=y, hue=hue, ax=ax, **kwargs)
    ax.set_xlabel(x)
    start = int(data[x].unique().min())
    stop = int(data[x].unique().max())
    if shade > 0:
        # xlim = ax.get_xlim()
        for x in np.arange(start, stop + 1, 2):
            ax.axvspan(x - 0.5, x + 0.5, color="lightgrey", alpha=0.2, linewidth=0)
    legend_upper_right(ax)
    return ax
