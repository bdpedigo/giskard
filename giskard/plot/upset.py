import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import seaborn as sns

# upset_ax = divider.append_axes("bottom", size=f"{upset_ratio*100}%", sharex=ax)


class UpsetCatplot:
    def __init__(self, fig, ax, upset_ax):
        self.fig = fig
        self.ax = ax
        self.upset_ax = upset_ax

    def set_ylabel(self, label, **kwargs):
        self.ax.set_ylabel(label, **kwargs)

    # TODO: allow for passing a dictionary to map old to new in case order changed
    def set_upset_ticklabels(self, ticklabels, **kwargs):
        self.upset_ax.set_yticklabels(ticklabels, **kwargs)

    def set_title(self, title, **kwargs):
        self.ax.set_title(title, **kwargs)


def upset_catplot(
    data,
    x=None,
    y=None,
    hue=None,
    kind="bar",
    estimator=None,
    estimator_width=0.2,
    estimator_labels=False,
    estimator_format="{estimate:.2f}",
    upset_ratio=0.3,
    upset_pad=0.7,
    upset_size=None,
    upset_linewidth=3,
    ax=None,
    figsize=(8, 6),
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    data = data.copy()
    groupby = data.groupby(x, sort=False)

    # combos = groupby.groups.keys()
    # combos = pd.DataFrame(combos, columns=x).set_index(x)

    # create a dummy variable for seaborn-style plotting
    group_estimates = []
    data["all_x_vars"] = np.nan
    for combo, index in groupby.groups.items():
        data.loc[index, "all_x_vars"] = str(combo)
        if estimator is not None:
            group_estimates.append(estimator(data.loc[index, y]))
    if kind == "bar":
        sns.barplot(data=data, x="all_x_vars", y=y, hue=hue, ax=ax, **kwargs)
    elif kind == "strip":
        sns.stripplot(data=data, x="all_x_vars", y=y, hue=hue, ax=ax, **kwargs)

    # TODO : could add other seaborn "kind"s

    # TODO : possibly more control over how this gets plotted
    # E.g. right now this would look ugly with barplot
    if estimator is not None:
        for i, estimate in enumerate(group_estimates):
            x_low = i - estimator_width
            x_high = i + estimator_width
            ax.plot([x_low, x_high], [estimate, estimate], color="black")
            if estimator_labels:
                pad = 0.02
                ax.text(
                    x_high + pad,
                    estimate,
                    estimator_format.format(estimate=estimate),
                    va="center",
                )

    ax.set(xlabel="")

    divider = make_axes_locatable(ax)
    upset_ax = divider.append_axes(
        "bottom", size=f"{upset_ratio*100}%", sharex=ax, pad=0
    )
    combos = data.set_index(x)
    plot_upset_indicators(
        combos,
        ax=upset_ax,
        height_pad=upset_pad,
        element_size=upset_size,
        linewidth=upset_linewidth,
    )

    return UpsetCatplot(fig, ax, upset_ax)


def plot_upset_indicators(
    intersections,
    ax=None,
    facecolor="black",
    element_size=None,
    with_lines=True,
    horizontal=True,
    height_pad=0.7,
    linewidth=2,
):
    # REF: https://github.com/jnothman/UpSetPlot/blob/e6f66883e980332452041cd1a6ba986d6d8d2ae5/upsetplot/plotting.py#L428
    """Plot the matrix of intersection indicators onto ax"""
    data = intersections
    index = data.index
    index = index.reorder_levels(index.names[::-1])
    n_cats = index.nlevels

    idx = np.flatnonzero(index.to_frame()[index.names].values)  # [::-1]
    c = np.array(["lightgrey"] * len(data) * n_cats, dtype="O")
    c[idx] = facecolor
    x = np.repeat(np.arange(len(data)), n_cats)
    y = np.tile(np.arange(n_cats), len(data))
    if element_size is not None:
        s = (element_size * 0.35) ** 2
    else:
        # TODO: make s relative to colw
        s = 200
    ax.scatter(x, y, c=c.tolist(), linewidth=0, s=s)

    if with_lines:
        line_data = (
            pd.Series(y[idx], index=x[idx]).groupby(level=0).aggregate(["min", "max"])
        )
        ax.vlines(
            line_data.index.values,
            line_data["min"],
            line_data["max"],
            lw=linewidth,
            colors=facecolor,
        )

    tick_axis = ax.yaxis
    tick_axis.set_ticks(np.arange(n_cats))
    tick_axis.set_ticklabels(index.names, rotation=0 if horizontal else -90)
    # ax.xaxis.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    if not horizontal:
        ax.yaxis.set_ticks_position("top")
    ax.set_frame_on(False)
    ax.set_ylim((-height_pad, n_cats - 1 + height_pad))
    ax.set_xticks([])
