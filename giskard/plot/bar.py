import pandas as pd
import matplotlib.pyplot as plt

# REF: https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_style_reference.html
hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]


def stacked_barplot(
    data: pd.Series,
    center=0,
    thickness=0.5,
    start=0,
    orient="v",
    ax=None,
    palette="deep",
    outline=False,
    colors=True,
    hatch_map=None,
    **kwargs
):
    """Draws a single stacked barplot based on a series of counts.

    Parameters
    ----------
    data : pd.Series
        Data to plot as a stacked bar, interpreted as counts.
    center : int or float (default=0)
        The center of the bar in the non-informative dimension - that is, the x axis if
        ``orient`` is "v" and the y axis if orient is "h".
    thickness : int or float (default=0.5)
        The width of the bar in the non-informative dimension.
    orient : str (default="v")
        The orientation of the bars, "v" for vertical, "h" for horizontal
    ax : matplotlib.axes.Axes, optional
        Axis on which to draw.
    palette : str, optional
        [description], by default "deep"
    start : int (default=0)
        Start of the bar in the informative dimension
    outline : bool, (default-False)
        Whether to outline the whole group of stacked bars. Can be useful if multiple
        stacked barplots will be drawn.
    """
    curr_start = start

    if orient == "v":
        drawer = ax.bar
    elif orient == "h":
        drawer = ax.barh

    for item, count in data.iteritems():
        if colors == True:
            color = palette[item]
        else:
            color = colors

        if hatch_map is not None:
            hatch = hatch_map[item]
        else:
            hatch = None
        drawer(
            center,
            count,
            thickness,
            curr_start,
            color=color,
            zorder=2,
            hatch=hatch,
            **kwargs
        )
        curr_start += count

    if outline:
        # TODO need to make this better at actually outlining the bar, right now the
        # spacing is kind of off
        drawer(
            center,
            data.sum(),
            thickness,
            start,
            edgecolor="black",
            linewidth=1,
            color="none",
            zorder=1,
        )


def crosstabplot(
    data: pd.DataFrame,
    group=None,
    group_order=None,
    group_order_aggfunc="mean",
    group_order_ascending=False,
    hue=None,
    hue_order=None,
    normalize=False,
    figsize=(8, 6),
    ax=None,
    palette=None,
    thickness=0.5,
    shift=0,
    orient="v",
    **kwargs
):
    counts_by_group = pd.crosstab(data[group], data[hue])
    if group_order is not None:
        if isinstance(group_order, str):
            if group_order_aggfunc == "mean":
                group_order_aggfunc = pd.Series.mean
            elif group_order_aggfunc == "mode":
                group_order_aggfunc = pd.Series.mode

            group_order = (
                data.groupby(group)[group_order]
                .agg(group_order_aggfunc)
                .sort_values(ascending=group_order_ascending)
                .index
            )
        elif isinstance(group_order, list):
            group_order = (
                data.groupby(group)[group_order]
                .mean()
                .sort_values(ascending=False)
                .index
            )
        counts_by_group = counts_by_group.reindex(index=group_order)

    if hue_order is not None:
        if isinstance(hue_order, str):
            hue_order = (
                data.groupby(hue)[hue_order].mean().sort_values(ascending=True).index
            )
        counts_by_group = counts_by_group.reindex(columns=hue_order)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    for i, (idx, row) in enumerate(counts_by_group.iterrows()):
        if normalize:
            row /= row.sum()
        stacked_barplot(
            row,
            center=i + shift,
            ax=ax,
            palette=palette,
            thickness=thickness,
            orient=orient,
            **kwargs
        )
    ax.set(xlabel=group, ylabel="Count")
    return ax
