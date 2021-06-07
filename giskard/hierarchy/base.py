from abc import abstractmethod

import numpy as np
import pandas as pd
from anytree import NodeMixin
from sklearn.base import BaseEstimator


class BaseNetworkTree(NodeMixin, BaseEstimator):
    def __init__(
        self,
        min_split=32,
        max_levels=2,
        verbose=False,
    ):
        self.min_split = min_split
        self.max_levels = max_levels
        self.verbose = verbose

    @property
    def node_data(self):
        if self.is_root:
            return self._node_data
        else:
            return self.root.node_data.loc[self._index]

    def fit(self, adjacency, node_data=None):
        self._check_node_data(adjacency, node_data)

        if self._check_continue_splitting(adjacency):
            if self.verbose > 0:
                print(f"{self._get_message_header(adjacency)} Splitting subgraph...")
            partition_labels = self._fit_partition(adjacency)
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
