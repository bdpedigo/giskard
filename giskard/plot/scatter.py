import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from graspologic.embed import select_dimension


def simple_scatterplot(
    X, labels=None, palette="deep", ax=None, title="", legend=False, figsize=(10, 10)
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plot_df = pd.DataFrame(data=X[:, :2], columns=["0", "1"])
    plot_df["labels"] = labels
    sns.scatterplot(data=plot_df, x="0", y="1", hue="labels", palette=palette, ax=ax)
    ax.set(xlabel="", ylabel="", title=title, xticks=[], yticks=[])
    ax.get_legend().remove()
    if legend:
        # convenient default that I often use, places in the top right outside of plot
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
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
