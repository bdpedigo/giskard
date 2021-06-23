import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def confusionplot(
    data1,
    data2=None,
    ax=None,
    figsize=(10, 10),
    xlabel="",
    ylabel="",
    title="Confusion matrix",
    annot=True,
    add_diag_proportion=True,
    normalize=None,
    return_confusion_matrix=False,
    **kwargs,
):
    if data1.ndim == 1:
        unique_labels = np.unique(list(data1) + list(data2))
        conf_mat = confusion_matrix(
            data1, data2, labels=unique_labels, normalize=normalize
        )
        conf_mat = pd.DataFrame(
            data=conf_mat, index=unique_labels, columns=unique_labels
        )
    else:
        conf_mat = data1

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(
        conf_mat,
        ax=ax,
        square=True,
        cmap="RdBu_r",
        center=0,
        cbar_kws=dict(shrink=0.6),
        annot=annot,
        fmt="d",
        mask=conf_mat == 0,
        **kwargs,
    )
    ax.set(ylabel=ylabel, xlabel=xlabel)

    if title != False:
        if add_diag_proportion:
            on_diag = np.trace(conf_mat.values) / np.sum(conf_mat.values)
            title += f" ({on_diag:0.2f} correct)"
        ax.set_title(title, fontsize="large", pad=10)
    if return_confusion_matrix:
        return ax, conf_mat
    return ax
