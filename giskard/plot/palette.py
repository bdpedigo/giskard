import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba_array
from .utils import spines_off


def palplot(
    p,
    n_colors=10,
    labels=None,
    figsize=(1, 10),
    ax=None,
    start=0,
    stop=None,
    tick_right=True,
):
    if isinstance(p, str):
        colors = sns.color_palette(palette=p, n_colors=n_colors)
        labels = np.arange(n_colors)
    elif isinstance(p, dict):
        colors = []
        if labels is None:
            keys = []
        for key, color in p.items():
            colors.append(color)
            if labels is None:
                keys.append(key)
        if labels is None:
            labels = keys
    elif isinstance(p, list):
        colors = p
        labels = np.arange(len(colors))
    n_colors = len(colors)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = np.array(colors)
    if colors.shape[-1] != 3:
        colors = to_rgba_array(colors)
        # colors = colors.reshape((n_colors, 1, 3))
        colors = colors.reshape((n_colors, 1, 4))
    else:
        colors = colors.reshape((n_colors, 1, 3))
    ax.imshow(colors)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.FixedLocator(np.arange(n_colors)))
    ax.yaxis.set_major_formatter(plt.FixedFormatter(labels))
    if tick_right:
        ax.yaxis.tick_right()
    ax.tick_params(length=0)
    spines_off(ax)
    return ax
