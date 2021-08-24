from abc import abstractmethod

import numpy as np
import pandas as pd
from anytree import NodeMixin, PreOrderIter, Walker
from sklearn.base import BaseEstimator


class BaseNetworkTree(NodeMixin, BaseEstimator):
    def __init__(self, min_split=32, max_levels=2, verbose=False, loops=False):
        self.min_split = min_split
        self.max_levels = max_levels
        self.verbose = verbose
        self.loops = loops

    @property
    def node_data(self):
        if self.is_root:
            return self._node_data
        else:
            return self.root.node_data.loc[self._index]

    def fit(self, adjacency, node_data=None):
        self._check_node_data(adjacency, node_data)

        partition_labels = self._fit_partition(adjacency)  # TODO whether to do prior or
        # after the check_continue_splitting
        if self._check_continue_splitting(adjacency):
            if self.verbose > 0:
                print(f"{self._get_message_header(adjacency)} Splitting subgraph...")

            self._split(adjacency, partition_labels)

        return self

    def _check_node_data(self, adjacency, node_data=None):
        if node_data is None and self.is_root:
            node_data = pd.DataFrame(index=range(adjacency.shape[0]))
            node_data["adjacency_index"] = range(adjacency.shape[0])
        if self.is_root:
            self._node_data = node_data
            self._index = node_data.index

    def _get_message_header(self, adjacency):
        return f"[Depth={self.depth}, Number of nodes={adjacency.shape[0]}]"

    def _check_continue_splitting(self, adjacency):
        return adjacency.shape[0] >= self.min_split and self.depth < self.max_levels

    def _split(self, adjacency, partition_labels):
        index = self._index
        node_data = self.root.node_data
        label_key = f"labels_{self.depth}"
        if label_key not in node_data.columns:
            node_data[label_key] = pd.Series(
                data=len(node_data) * [None], dtype="Int64"
            )

        unique_labels = np.unique(partition_labels)
        if self.verbose > 0:
            print(
                f"{self._get_message_header(adjacency)} Split into {len(unique_labels)} groups"
            )
        if len(unique_labels) > 1:
            for i, label in enumerate(unique_labels):
                mask = partition_labels == label
                sub_adjacency = adjacency[np.ix_(mask, mask)]
                self.root.node_data.loc[index[mask], f"labels_{self.depth}"] = i
                child = self.__class__(**self.get_params())
                child.parent = self
                child._index = index[mask]
                child.fit(sub_adjacency)

    @abstractmethod
    def _fit_partition(self, adjacency):
        """
        This method determines how to partition a given (sub)graph into smaller pieces.

        It should take in an adjacency matrix and return a n-length array which
        represents the partition.
        """
        pass

    def flatten_labels(self, level=None):
        pass

    def _hierarchical_mean(self, key):
        # TODO more generally, I think there's a place for "hierarchical agg"
        if self.is_leaf:
            index = self.node_data.index
            var = self.root.node_data.loc[index, key]
            return np.mean(var)
        else:
            children = self.children
            child_vars = [child._hierarchical_mean(key) for child in children]
            return np.mean(child_vars)

    def estimate_parameters(self, adjacency):
        # blah blah blah
        self._estimate_parameters(adjacency)
        return self

    def _aggregate_edges(self, adjacency):
        if isinstance(adjacency, np.ndarray):
            probability_estimate = np.count_nonzero(adjacency) / adjacency.shape[0]
        else:
            # I had a hunch the .count_nonzero method might be faster than the numpy
            # function but I could be totally wrong
            probability_estimate = adjacency.count_nonzero() / adjacency.shape[0]
        return probability_estimate

    def _estimate_parameters(self, adjacency):
        mask_arr = np.zeros(adjacency.shape[0], dtype=bool)
        indices = self.node_data["adjacency_index"]
        mask_arr[indices] = True
        mask = mask_arr[:, None] & mask_arr[None, :]
        if self.is_leaf and not self.loops:
            mask[indices, indices] = False
        else:
            counted = mask.copy()
            for child in self.children:
                child_mask = child._estimate_parameters(adjacency)
                mask[child_mask] = False
                counted[child_mask] = True
        probability_estimate = self._aggregate_edges(adjacency[mask])
        self.probability_estimate_ = probability_estimate
        if self.is_leaf and not self.loops:
            mask[indices, indices] = True
            return mask
        else:
            return counted

    @property
    def full_probability_matrix(self):
        n = len(self._index)
        data = np.empty((n, n))
        probability_matrix = pd.DataFrame(
            index=self._index, columns=self._index, data=data
        )
        for source_node in PreOrderIter(self):
            for target_node in PreOrderIter(self):
                nearest_common_ancestor = _get_nearest_common_ancestor(
                    source_node, target_node
                )
                probability = nearest_common_ancestor.probability_estimate_
                probability_matrix.loc[
                    source_node._index, target_node._index
                ] = probability
        return probability_matrix

    # @property
    # TODO need some way of naming things
    # def condensed_probability_matrix(self):
    #     n = len(self.descendants) + 1
    #     data = np.empty((n, n))
    #     probability_matrix = pd.DataFrame(
    #         index=self._index, columns=self._index, data=data
    #     )


def _get_nearest_common_ancestor(source, target):
    walker = Walker()
    _, nearest_common_ancestor, _ = walker.walk(source, target)
    return nearest_common_ancestor


class MetaTree(BaseNetworkTree):
    def __init__(self, min_split=32, max_levels=2, verbose=False, loops=False):
        super().__init__(
            min_split=min_split, max_levels=max_levels, verbose=verbose, loops=loops
        )

    def _check_continue_splitting(self):
        return len(self._index) >= self.min_split and self.depth < self.max_levels

    @property
    def size(self):
        return len(self._index)

    def build(self, node_data, prefix="", postfix="", offset=0):
        if self.is_root and ("adjacency_index" not in node_data.columns):
            node_data = node_data.copy()
            node_data["adjacency_index"] = range(len(node_data))
        self._index = node_data.index
        self._node_data = node_data
        key = prefix + f"{self.depth+offset}" + postfix
        if key in node_data and self._check_continue_splitting():
            if node_data[key].nunique() > 1:
                groups = node_data.groupby(key, dropna=False)
                for name, group in groups:
                    child = self.__class__(**self.get_params())
                    child.name = name
                    child.parent = self
                    child._index = group.index
                    child.build(group, prefix=prefix, postfix=postfix, offset=offset)
