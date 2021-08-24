import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anytree import LevelGroupOrderIter, NodeMixin, PreOrderIter
from sklearn.base import BaseEstimator

from .bar import stacked_barplot
from .utils import soft_axis_off


class MetaTree(NodeMixin, BaseEstimator):
    def __init__(self, min_split=0, max_levels=np.inf, verbose=False):
        self.min_split = min_split
        self.max_levels = max_levels
        self.verbose = verbose

    @property
    def node_data(self):
        if self.is_root:
            return self._node_data
        else:
            return self.root.node_data.loc[self._index]

    @property
    def size(self):
        return len(self._index)

    def _check_node_data(self, adjacency, node_data=None):
        if node_data is None and self.is_root:
            node_data = pd.DataFrame(index=range(adjacency.shape[0]))
        if self.is_root:
            self._node_data = node_data
            self._index = node_data.index

    def flatten_labels(self, level=None):
        pass

    def _hierarchical_mean(self, key):
        if self.is_leaf:
            index = self.node_data.index
            var = self.root.node_data.loc[index, key]
            return np.mean(var)
        else:
            children = self.children
            child_vars = [child._hierarchical_mean(key) for child in children]
            return np.mean(child_vars)

    def _hierarchical_mean_attribute(self, key):
        if self.is_leaf:
            return self.__getattribute__(key)
        else:
            children = self.children
            child_vars = [child._hierarchical_mean_attribute(key) for child in children]
            return np.mean(child_vars)

    def _check_continue_splitting(self):
        return len(self._index) >= self.min_split and self.depth < self.max_levels

    def build(self, node_data, prefix="", postfix=""):
        # if self.is_root and ("adjacency_index" not in node_data.columns):
        #     node_data = node_data.copy()
        #     node_data["adjacency_index"] = range(len(node_data))
        self._index = node_data.index
        self._node_data = node_data
        key = prefix + f"{self.depth}" + postfix
        if key in node_data.columns and self._check_continue_splitting():
            groups = node_data.groupby(key)
            for _, group_data in groups:
                child = MetaTree(min_split=self.min_split, max_levels=self.max_levels)
                child.parent = self
                child._index = group_data.index
                child.build(group_data, prefix=prefix, postfix=postfix)

    def order(self, order):
        for node in PreOrderIter(self):
            if not node.is_leaf:
                children = np.array(node.children)
                children_data = [n.node_data for n in children]
                priority_values = []
                for child_data in children_data:
                    val = child_data[order].mean()
                    priority_values.append(val)
                inds = np.argsort(priority_values)
                node.children = tuple(children[inds])

    def sort_values(self, by, *args, **kwargs):
        kwargs["inplace"] = True
        self.node_data.sort_values(by, *args, **kwargs)


# def _dendrogram_leaf_aligned(tree):
# TODO

# def _dendrogram_repeated(tree):
# TODO


def _dendrogram_root_aligned(
    tree, ax, hue=None, orient="h", palette=None, hue_order_index=None, thickness=0.5
):
    for i, group in enumerate(LevelGroupOrderIter(tree)):
        for node in group:
            center = node._hierarchical_mean_attribute("center_span")
            node.center_span = center
            node.center_extent = i
            counts = node.node_data[hue].value_counts()
            counts = counts.reindex(hue_order_index).dropna()
            start = center - node.size / 2
            stacked_barplot(
                counts,
                center=i,
                start=start,
                orient=orient,
                ax=ax,
                palette=palette,
                thickness=thickness,
            )


def get_x_y(xs, ys, orient):
    if orient == "h":
        return xs, ys
    elif orient == "v":
        return (ys, xs)


def _draw_line(spans, extents, ax=None, orient="h", linewidth=1, color="black"):
    xs, ys = get_x_y(spans, extents, orient=orient)
    ax.plot(xs, ys, linewidth=linewidth, color=color, alpha=1)


def _draw_connector_lines(tree, orient="h", linewidth=1, ax=None, thickness=0.5):
    for node in PreOrderIter(tree):
        if not node.is_leaf:
            children = node.children
            min_span = min([child.center_span for child in children])
            max_span = max([child.center_span for child in children])
            mean_span = (min_span + max_span) / 2
            current_bottom = node.center_extent + thickness / 2
            next_top = node.center_extent + 1 - thickness / 2
            middle = (current_bottom + next_top) / 2

            spans = [mean_span, mean_span]
            extents = [current_bottom, middle]
            _draw_line(spans, extents, ax=ax, orient=orient, linewidth=linewidth)

            spans = [min_span, max_span]
            extents = [middle, middle]
            _draw_line(spans, extents, ax=ax, orient=orient, linewidth=linewidth)

            extents = [middle, next_top]
            for child in children:
                spans = [child.center_span, child.center_span]
                _draw_line(spans, extents, ax=ax, orient=orient, linewidth=linewidth)


def dendrogram_barplot(
    data,
    group=None,
    hue=None,
    group_order=None,
    hue_order=None,
    max_levels=np.inf,
    align="root",
    min_split=0,
    orient="h",
    pad=10,
    ax=None,
    figsize=(10, 2),
    palette=None,
    thickness=0.5,
    linewidth=1,
):
    data = data.copy()

    # if group_order is not None:
    #     group_order_vals = data.groupby()[group_order].mean()
    #     group_order_vals = group_order_vals.sort_values()
    #     group_order_cat = pd.Categorical(
    #         data[group], categories=group_order_vals, ordered=True
    #     )
    #     data['_group'] = group_order_cat
    #     data.sort_values('_group', inplace=True)

    tree = MetaTree(min_split=min_split, max_levels=max_levels)
    tree.build(data, prefix=group)
    if group_order is not None:
        tree.order(group_order)
    if hue_order is not None:
        tree.sort_values(hue_order)

    mean_vals = data.groupby(hue)[hue_order].mean()
    if isinstance(mean_vals, pd.Series):
        mean_vals = mean_vals.to_frame()
    mean_vals.sort_values(hue_order, inplace=True)
    hue_order_index = mean_vals.index

    cumulative_span = 0
    for i, leaf in enumerate(tree.leaves):
        leaf.leaf_id = i
        leaf.center_span = cumulative_span + leaf.size / 2
        cumulative_span += leaf.size + pad

    _, ax = plt.subplots(1, 1, figsize=figsize)
    # ax.invert_yaxis()
    ax.set_ylim((cumulative_span, 0))

    if palette is None:
        colors = cc.glasbey_light
        vals = np.unique(data[hue])
        palette = dict(zip(vals, colors))

    _dendrogram_root_aligned(
        tree,
        hue=hue,
        ax=ax,
        orient=orient,
        palette=palette,
        hue_order_index=hue_order_index,
    )

    _draw_connector_lines(
        tree, orient=orient, ax=ax, thickness=thickness, linewidth=linewidth
    )

    soft_axis_off(ax)

    return ax
