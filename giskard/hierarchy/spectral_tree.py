from .base import BaseNetworkTree


class SpectralTree(BaseNetworkTree):
    def __init__(
        self,
        verbose=False,
        min_split=32,
        max_levels=2,
        n_components=None,
        embed_algorithm="ase",
        cluster_algorithm="gmm",
        embed_kws={},
        cluster_kws={},
    ):
        super().__init__(
            min_split=min_split,
            max_levels=max_levels,
            verbose=verbose,
        )

    def _fit_partition(self, adjacency):
        """Fits a partition to the current subgraph using spectral clustering"""
        pass
        # return partition_labels
