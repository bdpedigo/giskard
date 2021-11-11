from .base import BaseNetworkTree
from graspologic.partition import leiden
import numpy as np


class LeidenTree(BaseNetworkTree):
    def __init__(
        self,
        trials=1,
        resolution=1.0,
        min_split=32,
        max_levels=2,
        verbose=False,
    ):
        super().__init__(
            min_split=min_split,
            max_levels=max_levels,
            verbose=verbose,
        )
        self.trials = trials
        self.resolution = resolution

    def _fit_partition(self, adjacency):
        """Fits a partition to the current subgraph using Leiden"""
        partition_map = leiden(adjacency, trials=self.trials)
        partition_labels = np.vectorize(partition_map.__getitem__)(
            np.arange(adjacency.shape[0])
        )
        return partition_labels

    #TODO can think about modifications to _check_continue_splitting