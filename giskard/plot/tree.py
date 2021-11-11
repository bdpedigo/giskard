import anytree
from matplotlib.patches import Circle


def get_x_y(xs, ys, orientation):
    if orientation == "h":
        return xs, ys
    elif orientation == "v":
        return (ys, xs)


def plot_dendrogram(
    ax,
    root: anytree.AnyNode,
    index_key="sorted_adjacency_index",
    orientation="h",
    linewidth=0.7,
    cut=None,
    lowest_level=None,
    markersize=None,
    markercolor="b",
    linecolor="black",
    fontsize="medium",
):
    if lowest_level is None:
        lowest_level = root.height

    for node in (root.descendants) + (root,):
        y = node._hierarchical_mean(index_key)
        x = node.depth
        node.y = y + 0.5
        node.x = x

    walker = anytree.Walker()
    walked = []

    for node in root.leaves:
        upwards, _, _ = walker.walk(node, root)
        curr_node = node
        for up_node in (upwards) + (root,):
            edge = (curr_node, up_node)
            if edge not in walked:
                xs = [curr_node.x, up_node.x]
                ys = [curr_node.y, up_node.y]
                xs, ys = get_x_y(xs, ys, orientation)
                ax.plot(
                    xs,
                    ys,
                    linewidth=linewidth,
                    color=linecolor,
                    alpha=1,
                )
                walked.append(edge)
            curr_node = up_node
        y_max = node.node_data[index_key].max() + 1
        y_min = node.node_data[index_key].min()
        # TODO maybe don't hard-code some of these values
        # TODO give a few options for visual presentation of the hierarchies
        # For instance, the lines drawn in the dendrogram could all be axis-aligned
        triangle_pad = 0
        xs = [node.x, node.x, node.x + 1, node.x + 1]
        ys = [node.y - triangle_pad, node.y + triangle_pad, y_max, y_min]
        xs, ys = get_x_y(xs, ys, orientation)
        ax.fill(xs, ys, facecolor=linecolor)

    if orientation == "h":
        ax.set(xlim=(-1, lowest_level + 1))
        if cut is not None:
            ax.axvline(cut - 1, linewidth=1, color="grey", linestyle=":")
    elif orientation == "v":
        ax.set(ylim=(lowest_level + 1, -1))
        if cut is not None:
            ax.axhline(cut - 1, linewidth=1, color="grey", linestyle=":")

    if markersize is not None:
        for node in (root.descendants) + (root,):
            if hasattr(node, "name"):
                name = node.name
            else:
                name = ""
            x = node.x
            y = node.y
            x, y = get_x_y(x, y, orientation)
            ax.plot(x, y, marker="o", color=markercolor, markersize=markersize, linewidth=1, markeredgecolor='black')
            ax.text(x, y, name, ha="center", va="center", fontsize=fontsize)
    ax.axis("off")


# TODO bar dendrogram

# TODO treeplot
