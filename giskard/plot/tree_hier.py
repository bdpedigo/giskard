import math
import random
from collections import namedtuple

import anytree
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from giskard.plot import soft_axis_off


def hierarchy_pos(
    G,
    root=None,
    width=1.0,
    vert_gap=0.2,
    vert_loc=0,
    leaf_vs_root_factor=0.5,
    xcenter=0.5,
):
    # REF
    # https://epidemicsonnetworks.readthedocs.io/en/latest/_modules/EoN/auxiliary.html#hierarchy_pos
    """
    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).

    There are two basic approaches we think of to allocate the horizontal
    location of a node.

    - Top down: we allocate horizontal space to a node.  Then its ``k``
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.

    We use use both of these approaches simultaneously with ``leaf_vs_root_factor``
    determining how much of the horizontal space is based on the bottom up
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.


    :Arguments:

    **G** the graph (must be a tree)

    **root** the root node of the tree
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root

    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
        G,
        root,
        leftmost,
        width,
        leafdx=0.2,
        vert_gap=0.2,
        vert_loc=0,
        xcenter=0.5,
        rootpos=None,
        leafpos=None,
        parent=None,
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if rootpos is None:
            rootpos = {root: (xcenter, vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            rootdx = width / len(children)
            nextx = xcenter - width / 2 - rootdx / 2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos(
                    G,
                    child,
                    leftmost + leaf_count * leafdx,
                    width=rootdx,
                    leafdx=leafdx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    rootpos=rootpos,
                    leafpos=leafpos,
                    parent=root,
                )
                leaf_count += newleaves

            leftmostchild = min((x for x, y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x, y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild + rightmostchild) / 2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root] = (leftmost, vert_loc)
        return rootpos, leafpos, leaf_count

    xcenter = width / 2.0
    if isinstance(G, nx.DiGraph):
        leafcount = len(
            [node for node in nx.descendants(G, root) if G.out_degree(node) == 0]
        )
    elif isinstance(G, nx.Graph):
        leafcount = len(
            [
                node
                for node in nx.node_connected_component(G, root)
                if G.degree(node) == 1 and node != root
            ]
        )
    rootpos, leafpos, leaf_count = _hierarchy_pos(
        G,
        root,
        0,
        width,
        leafdx=width * 1.0 / leafcount,
        vert_gap=vert_gap,
        vert_loc=vert_loc,
        xcenter=xcenter,
    )
    pos = {}
    for node in rootpos:
        pos[node] = (
            leaf_vs_root_factor * leafpos[node][0]
            + (1 - leaf_vs_root_factor) * rootpos[node][0],
            leafpos[node][1],
        )
    xmax = max(x for x, y in pos.values())
    for node in pos:
        pos[node] = (pos[node][0] * width / xmax, pos[node][1])
    return pos


def construct_tree_graph(root, max_depth=np.inf):
    tree_g = nx.DiGraph()
    for node in anytree.PreOrderIter(root):
        tree_g.add_node(node.id, n_reports=len(node.descendants) + 1, depth=node.depth)
        for child in node.children:
            tree_g.add_edge(node.id, child.id)
    return tree_g


TreeplotResult = namedtuple("TreeplotResult", ["ax", "nodelist", "pos"])


def treeplot(
    tree_g,
    size_threshold=50,
    layout="radial",
    node_size_scale=5,
    node_hue=None,
    edge_hue=None,
    figsize=(10, 10),
    ax=None,
    edge_linewidth=1,
    node_palette=None,
    edge_palette=None,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    big_nodes = [
        x for x, y in tree_g.nodes(data=True) if y["n_reports"] >= size_threshold
    ]
    big_tree_g = tree_g.subgraph(big_nodes)

    nodelist = sorted(big_tree_g.nodes)
    edgelist = sorted(big_tree_g.edges)

    if layout == "radial":
        pos = hierarchy_pos(big_tree_g, width=2 * math.pi, xcenter=0)
        pos = {
            u: (r * math.cos(theta), r * math.sin(theta))
            for u, (theta, r) in pos.items()
        }
    elif layout == "vertical":
        pos = hierarchy_pos(big_tree_g)

    node_size = [
        node_size_scale * (big_tree_g.nodes[node]["n_reports"] - 1) for node in nodelist
    ]

    if node_hue is not None:
        if node_hue == "id":
            node_hues = nodelist
        else:
            node_hues = [tree_g.nodes[n][node_hue] for n in nodelist]

    if node_palette is None and node_hue is not None:
        node_palette = dict(zip(np.unique(node_hues), sns.color_palette("tab10")))

    if edge_hue is not None:
        edge_hues = [tree_g.edges[u, v][edge_hue] for u, v in edgelist]
        edge_color = list(map(edge_palette.__getitem__, edge_hues))
    else:
        edge_color = "black"

    node_color = list(map(node_palette.__getitem__, node_hues))
    nx.draw_networkx(
        big_tree_g,
        pos=pos,
        with_labels=False,
        nodelist=nodelist,
        edgelist=edgelist,
        node_size=node_size,
        ax=ax,
        arrows=False,
        width=edge_linewidth,
        node_color=node_color,
        edge_color=edge_color,
    )

    soft_axis_off(ax)
    ax.axis("square")

    return TreeplotResult(ax, nodelist, pos)
