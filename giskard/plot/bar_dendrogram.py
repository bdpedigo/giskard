import numpy as np


def dendrogram_barplot(data):
    pass


def plot_single_dendrogram(meta, axs):
    n_leaf = meta[f"lvl{lowest_level}_labels"].nunique()
    # n_pairs = len(full_meta) // 2
    n_cells = len(meta)

    first_mid_map = get_mid_map(full_meta, bilat=True)

    # left side
    # meta = full_meta[full_meta["hemisphere"] == "L"].copy()
    ax = axs[0]
    ax.set_ylim((-gap, (n_cells + gap * n_leaf)))
    ax.set_xlim((-0.5, lowest_level + 2 + 0.5))

    draw_bar_dendrogram(meta, ax, first_mid_map)

    ax.set_yticks([])
    ax.set_xticks(np.arange(lowest_level + 1))
    ax.tick_params(axis="both", which="both", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Level")

    # add a scale bar in the bottom left
    ax.bar(x=0, height=100, bottom=0, width=width, color="k")
    ax.text(x=0.35, y=0, s="100 neurons")

    # ax = axs[1]
    # plot_color_labels(full_meta, ax)


def get_mid_map(full_meta, leaf_key=None, bilat=False):
    if leaf_key is None:
        leaf_key = f"lvl{lowest_level}_labels"
    # left
    if not bilat:
        meta = full_meta[full_meta["hemisphere"] == "L"].copy()
    else:
        meta = full_meta.copy()

    sizes = meta.groupby([leaf_key, "merge_class"], sort=False).size()

    uni_labels = sizes.index.unique(0)

    mids = []
    offset = 0
    for ul in uni_labels:
        heights = sizes.loc[ul]
        starts = heights.cumsum() - heights + offset
        offset += heights.sum() + gap
        minimum = starts[0]
        maximum = starts[-1] + heights[-1]
        mid = (minimum + maximum) / 2
        mids.append(mid)

    left_mid_map = dict(zip(uni_labels, mids))
    if bilat:
        first_mid_map = {}
        for k in left_mid_map.keys():
            left_mid = left_mid_map[k]
            first_mid_map[k + "-"] = left_mid
        return first_mid_map

    # right
    meta = full_meta[full_meta["hemisphere"] == "R"].copy()

    sizes = meta.groupby([leaf_key, "merge_class"], sort=False).size()

    # uni_labels = np.unique(labels)
    uni_labels = sizes.index.unique(0)

    mids = []
    offset = 0
    for ul in uni_labels:
        heights = sizes.loc[ul]
        starts = heights.cumsum() - heights + offset
        offset += heights.sum() + gap
        minimum = starts[0]
        maximum = starts[-1] + heights[-1]
        mid = (minimum + maximum) / 2
        mids.append(mid)

    right_mid_map = dict(zip(uni_labels, mids))

    keys = list(set(list(left_mid_map.keys()) + list(right_mid_map.keys())))
    first_mid_map = {}
    for k in keys:
        left_mid = left_mid_map[k]
        right_mid = right_mid_map[k]
        first_mid_map[k + "-"] = max(left_mid, right_mid)
    return first_mid_map


def draw_bar_dendrogram(
    meta,
    ax,
    first_mid_map,
    lowest_level=7,
    width=0.5,
    draw_labels=False,
    color_key="merge_class",
    color_order="sf",
):
    meta = meta.copy()
    last_mid_map = first_mid_map
    line_kws = dict(linewidth=1, color="k")
    for level in np.arange(lowest_level + 1)[::-1]:
        x = level
        # mean_in_cluster = meta.groupby([f"lvl{level}_labels", color_key])["sf"].mean()
        meta = meta.sort_values(
            [f"lvl{level}_labels", f"{color_key}_{color_order}_order", color_key],
            ascending=True,
        )
        sizes = meta.groupby([f"lvl{level}_labels", color_key], sort=False).size()

        uni_labels = sizes.index.unique(level=0)  # these need to be in the right order

        mids = []
        for ul in uni_labels:
            if not isinstance(ul, str):
                ul = str(ul)  # HACK
            last_mids = get_last_mids(ul, last_mid_map)
            grand_mid = np.mean(last_mids)

            heights, starts, colors = calc_bar_params(sizes, ul, grand_mid)

            minimum = starts[0]
            maximum = starts[-1] + heights[-1]
            mid = (minimum + maximum) / 2
            mids.append(mid)

            # draw the bars
            for i in range(len(heights)):
                ax.bar(
                    x=x,
                    height=heights[i],
                    width=width,
                    bottom=starts[i],
                    color=colors[i],
                )
                if (level == lowest_level) and draw_labels:
                    ax.text(
                        x=lowest_level + 0.5, y=mid, s=ul, verticalalignment="center"
                    )

            # draw a horizontal line from the middle of this bar
            if level != 0:  # dont plot dash on the last
                ax.plot([x - 0.5 * width, x - width], [mid, mid], **line_kws)

            # line connecting to children clusters
            if level != lowest_level:  # don't plot first dash
                ax.plot(
                    [x + 0.5 * width, x + width], [grand_mid, grand_mid], **line_kws
                )

            # draw a vertical line connecting the two child clusters
            if len(last_mids) == 2:
                ax.plot([x + width, x + width], last_mids, **line_kws)

        last_mid_map = dict(zip(uni_labels, mids))
    remove_spines(ax)
