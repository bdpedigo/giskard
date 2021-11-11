import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .utils import soft_axis_off

# data=None,
# row_meta=None,
# col_meta=None,
# plot_type="heatmap",
# col_group=None,  # single string, list of string, or np.ndarray
# row_group=None,  # can also represent a clustering?
# col_group_order="size",
# row_group_order="size",
# col_dendrogram=None,  # can this just be true false?
# row_dendrogram=None,
# col_item_order=None,  # single string, list of string, or np.ndarray
# row_item_order=None,
# col_colors=None,  # single string, list of string, or np.ndarray
# row_colors=None,
# col_palette="tab10",
# row_palette="tab10",


class MatrixGrid:
    def __init__(
        self,
        col_ticks=True,
        row_ticks=True,
        col_tick_pad=None,
        row_tick_pad=None,
        ax=None,
        figsize=(10, 10),
        gap=False,
        spines=True,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        else:
            fig = ax.figure
        divider = make_axes_locatable(ax)

        self.spines = spines
        self.gap = gap

        self.fig = fig
        self.ax = ax
        self.divider = divider
        self.top_axs = []
        self.left_axs = []
        self.bottom_axs = []
        self.right_axs = []
        self.side_axs = {
            "top": self.top_axs,
            "bottom": self.bottom_axs,
            "left": self.left_axs,
            "right": self.right_axs,
        }

        self._set_spines(spines)

    def _set_spines(self, spines):
        soft_axis_off(ax=self.ax, left=spines, right=spines, top=spines, bottom=spines)

    @property
    def all_top_axs(self):
        return [self.ax] + self.top_axs

    @property
    def all_bottom_axs(self):
        return [self.ax] + self.bottom_axs

    @property
    def all_left_axs(self):
        return [self.ax] + self.left_axs

    @property
    def all_right_axs(self):
        return [self.ax] + self.right_axs

    def append_axes(self, side, size="10%", pad=0):
        kws = {}
        if side in ["top", "bottom"]:
            kws["sharex"] = self.ax
        elif side in ["left", "right"]:
            kws["sharey"] = self.ax
        if self.spines and len(self.side_axs[side]) == 0:
            # HACK: usually need to add something small here so that the new axis isn't
            # covering the main plot's spines
            pad += 0.01
        ax = self.divider.append_axes(side, size=size, pad=pad, **kws)
        if side == "top":
            self.top_axs.append(ax)
        elif side == "bottom":
            self.bottom_axs.append(ax)
        elif side == "left":
            self.left_axs.append(ax)
        elif side == "right":
            self.right_axs.append(ax)
        return ax

    def set_title(self, title, **kwargs):
        for ax in self.all_top_axs:
            ax.set_title("", **kwargs)
        text = self.all_top_axs[-1].set_title(title, **kwargs)
        return text

    def set_xlabel(self, xlabel, **kwargs):
        for ax in self.all_bottom_axs:
            ax.set_xlabel("", **kwargs)
        text = self.all_bottom_axs[-1].set_xlabel(xlabel, **kwargs)
        return text

    def set_ylabel(self, ylabel, **kwargs):
        for ax in self.all_left_axs:
            ax.set_ylabel("", **kwargs)
        text = self.all_left_axs[-1].set_ylabel(ylabel, **kwargs)
        return text
