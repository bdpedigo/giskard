import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pairplot(X, labels=None, figsize=(10, 10), alpha=0.8, linewidth=0, s=10, **kwargs):
    # TODO currently has no legend support
    n_dims = X.shape[1]
    data = pd.DataFrame(X, columns=[f"{i}" for i in range(n_dims)])
    if labels is not None:
        data["hue"] = list(labels)
        hue = "hue"
    else:
        hue = None
    fig, axs = plt.subplots(n_dims, n_dims, figsize=figsize)
    for i in range(n_dims):
        for j in range(n_dims):
            ax = axs[i, j]
            if i < j:
                sns.scatterplot(
                    data=data,
                    x=f"{i}",
                    y=f"{j}",
                    hue=hue,
                    legend=False,
                    ax=ax,
                    linewidth=linewidth,
                    alpha=alpha,
                    s=s,
                    **kwargs,
                )
                ax.set(xticks=[], yticks=[], xlabel="", ylabel="")
            else:
                ax.axis("off")
    return fig, axs
