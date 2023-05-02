import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pairplot(
    X,
    labels=None,
    figsize=(10, 10),
    alpha=0.8,
    linewidth=0,
    s=10,
    subsample=None,
    title=None,
    title_y=0.93,
    title_fontsize="xx-large",
    marginals=False,
    **kwargs,
):
    # TODO currently has no legend support
    n_dims = X.shape[1]
    data = pd.DataFrame(X, columns=[f"{i}" for i in range(n_dims)])
    if labels is not None:
        data["hue"] = list(labels)
        hue = "hue"
    else:
        hue = None

    if subsample is not None:
        data = data.sample(frac=subsample, replace=False)

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
                    legend=True,
                    ax=ax,
                    linewidth=linewidth,
                    alpha=alpha,
                    s=s,
                    **kwargs,
                )
                if i == 0 and j == 1:
                    handles, labels = ax.get_legend_handles_labels()
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()
                ax.set(xticks=[], yticks=[], xlabel="", ylabel="")
            else:
                ax.axis("off")

    if marginals:
        for i in range(n_dims):
            ax = axs[i, i]
            sns.histplot(
                data=data,
                x=f"{i}",
                ax=ax,
                hue=hue,
                palette=kwargs.get("palette", None),
                legend=False,
            )
            ax.set(xticks=[], yticks=[], xlabel="", ylabel="")

    if marginals:
        ax = axs[1, 0]
    else:
        ax = axs[0, 0]
    ax.legend(handles, labels, loc="upper right", frameon=True)

    fig.suptitle(title, y=title_y, fontsize=title_fontsize)

    return fig, axs
