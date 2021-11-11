from .bar import stacked_barplot, crosstabplot
from .hist import histplot
from .matrix import dissimilarity_clustermap, plot_squarelines, scattermap
from .scatter import (
    simple_scatterplot,
    screeplot,
    textplot,
    simple_umap_scatterplot,
    matched_stripplot,
    scatterplot,
)
from .utils import (
    soft_axis_off,
    axis_on,
    legend_upper_right,
    merge_axes,
    remove_shared_ax,
    rotate_labels,
)
from .graph_layout import graphplot
from .confusion import confusionplot
from .matrixgrid import MatrixGrid
from .theme import set_theme
from .tree import plot_dendrogram
from .bar_dendrogram import dendrogram_barplot
from .pairplot import pairplot
from .old_matrixplot import adjplot, matrixplot
from .subuniformity import subuniformity_plot
