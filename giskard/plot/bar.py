import pandas as pd


def stacked_barplot(
    data: pd.Series,
    center=0,
    thickness=0.5,
    orient="v",
    ax=None,
    palette="deep",
    start=0,
    outline=False,
):
    """Draws a single stacked barplot based on a series of counts.

    Parameters
    ----------
    data : pd.Series
        Data to plot as a stacked bar, interpreted as counts
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
        drawer(center, count, thickness, curr_start, color=palette[item], zorder=2)
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
