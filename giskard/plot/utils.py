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