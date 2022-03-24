import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import colorcet as cc
import matplotlib as mpl


def soft_axis_off(ax, top=False, bottom=False, left=False, right=False):
    ax.set(xlabel="", ylabel="", xticks=[], yticks=[])
    ax.spines["top"].set_visible(top)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["left"].set_visible(left)
    ax.spines["right"].set_visible(right)


def axis_on(ax):
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(True)


def _get_slice(items, n_items):
    if items is None:
        return slice(0, n_items)
    elif type(items) == int:
        return items
    elif len(items) == 1:
        return items[0]
    else:
        return slice(items[0], items[1])


def merge_axes(fig, axs, rows=None, cols=None):
    # TODO I could't figure out a safer way to do this without eval
    # seems like gridspec.__getitem__ only wants numpy indices in the slicing form
    # REF: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_and_subplots.html
    row_slice = _get_slice(rows, axs.shape[0])
    col_slice = _get_slice(cols, axs.shape[1])
    gs = axs.flat[0].get_gridspec()
    for ax in axs[row_slice, col_slice].ravel():
        ax.remove()
    ax = fig.add_subplot(gs[row_slice, col_slice])
    return ax


def legend_upper_right(ax, **kwargs):
    ax.get_legend().remove()
    leg = ax.legend(bbox_to_anchor=(1, 1), loc="upper left", **kwargs)
    return leg


def make_palette(
    labels, palette=None, default=None, size_threshold=None, top_k_threshold=None
):
    # TODO capability to only color the top k categories
    # TODO pass back the palette if it exists
    # TODO optionally add a default value for all unmentioned/below threshold labels
    raise NotImplementedError()
    uni_labels, counts = np.unique(labels, return_counts=True)
    n_categories = len(uni_labels)
    if n_categories <= 10:
        colors = sns.color_palette("tab10")
    elif n_categories <= 20:
        colors = sns.color_palette("tab20")
    else:
        colors = cc.glasbey_light


def make_axes(ax=None, figsize=(8, 6)):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    return ax


def remove_shared_ax(ax, x=True, y=True):

    axes = []
    if x:
        shax = ax.get_shared_x_axes()
        shax.remove(ax)
        axes.append(ax.xaxis)
    if y:
        shay = ax.get_shared_y_axes()
        shay.remove(ax)
        axes.append(ax.yaxis)

    for axis in axes:
        ticker = mpl.axis.Ticker()
        axis.major = ticker
        axis.minor = ticker
        loc = mpl.ticker.NullLocator()
        fmt = mpl.ticker.NullFormatter()
        axis.set_major_locator(loc)
        axis.set_major_formatter(fmt)
        axis.set_minor_locator(loc)
        axis.set_minor_formatter(fmt)


def rotate_labels(ax):
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
