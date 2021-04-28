import numpy as np
import seaborn as sns


def histplot(data, x=None, hue=None, ax=None, side_legend=True, kde=False, **kwargs):
    sizes = data.groupby(hue).size()
    single_hues = np.unique(sizes[sizes == 1].index)
    single_data = data[data[hue].isin(single_hues)]
    other_data = data[~data[hue].isin(single_hues)]
    if kde:
        sns.kdeplot(
            data=other_data, x=x, hue=hue, ax=ax, legend=True, fill=True, **kwargs
        )
    else:
        sns.histplot(data=other_data, x=x, hue=hue, ax=ax, legend=True, **kwargs)
    # this was not working, unsure why
    # handles, labels = ax.get_legend_handles_labels()
    legend = ax.get_legend()
    handles = legend.legendHandles
    labels = legend.get_texts()
    labels = [label.get_text() for label in labels]
    for idx, row in single_data.iterrows():
        x_val = row[x]
        # TODO add palette support
        line = ax.axvline(x_val, color="darkred", linestyle="--", linewidth=2)
        handles.append(line)
        labels.append(row[hue])

    if side_legend:
        ax.get_legend().remove()
        ax.legend(
            handles=handles, labels=labels, bbox_to_anchor=(1, 1), loc="upper left"
        )
