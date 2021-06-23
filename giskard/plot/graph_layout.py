# %%
import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from graspologic.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspologic.utils import import_graph, pass_to_ranks
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from umap import UMAP


def graphplot(
    network=None,
    embedding=None,
    meta=None,
    transform="pass_to_ranks",
    embedding_algorithm="ase",
    n_components=32,
    n_neighbors=32,
    min_dist=0.8,
    metric="cosine",
    hue=None,
    group="hue",
    group_convex_hull=False,
    size="degree",
    node_palette=None,
    ax=None,
    figsize=(10, 10),
    sizes=(10, 30),
    legend=False,
    edge_hue="pre",
    edge_palette=None,
    edge_linewidth=0.2,
    edge_alpha=0.2,
    spines=False,
    subsample_edges=False,
    verbose=False,
    random_state=None,
    network_order=1,
    normalize_power=False,
    supervised_weight=False,
    hue_labels=False,
    hue_label_fontsize=None,
    adjust_labels=False,
    return_results=False,
    tile=False,
    tile_layout=None,
    embed_kws={},
    umap_kws={},
    scatterplot_kws={},
):
    results = {}

    networkx = False
    adj = import_graph(network).copy()  # TODO allow for CSR

    if random_state is None:
        random_state = np.random.default_rng()
    elif isinstance(random_state, (int, np.integer)):
        random_state = np.random.default_rng(random_state)

    if transform == "pass_to_ranks":
        adj = pass_to_ranks(adj)

    if embedding is None:
        # if we are given a graph, do an initial embedding

        if verbose > 0:
            print("Performing initial spectral embedding of the network...")
        if embedding_algorithm == "ase":
            embedder = AdjacencySpectralEmbed(
                n_components=n_components, concat=True, **embed_kws
            )
        elif embedding_algorithm == "lse":
            embedder = LaplacianSpectralEmbed(
                form="R-DAD", n_components=n_components, concat=True, **embed_kws
            )
        if network_order == 2:
            # TODO not sure how much this makes sense in practice, just something I've
            # been playing around with
            if normalize_power:
                adj_normed = normalize(adj, axis=1)
                embedding = embedder.fit_transform(adj_normed @ adj_normed)
            else:
                embedding = embedder.fit_transform(adj @ adj)
        elif network_order == 1:
            embedding = embedder.fit_transform(adj)

        results["embedding"] = embedding

    # if input is networkx, extract node metadata into a data frame
    if isinstance(network, (nx.Graph, nx.DiGraph)):
        networkx = True
        index = list(sorted(network.nodes()))
        meta = pd.DataFrame(index=index)
        for attr in [hue, size]:
            if attr is not None:
                attr_map = nx.get_node_attributes(network, attr)
                meta[attr] = meta.index.map(attr_map)
    elif meta is None:
        meta = pd.DataFrame(index=range(network.shape[0]))
    index = meta.index

    if embedding.shape[1] > 2:
        if verbose > 0:
            print("Performing UMAP embedding...")
        # once we have the initial embedding, embed again down to 2D using UMAP
        umapper = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state.integers(np.iinfo(np.int32).max),
            target_weight=supervised_weight,
            **umap_kws,
        )
        if supervised_weight > 0:
            if group == "hue":
                group = hue
            y = meta[group].values
            _, y = np.unique(y, return_inverse=True)
        else:
            y = None
        umap_embedding = umapper.fit_transform(embedding, y=y)
        results["umap_embedding"] = umap_embedding
    else:
        umap_embedding = embedding

    # TODO
    mids = (umap_embedding.max(axis=0) + umap_embedding.min(axis=0)) / 2
    umap_embedding -= mids
    max_length = np.linalg.norm(umap_embedding, axis=1).max()
    umap_embedding /= max_length

    # add the UMAP embedding into the dataframe for plotting
    columns = [f"umap_{i}" for i in range(umap_embedding.shape[1])]
    plot_df = pd.DataFrame(data=umap_embedding, columns=columns, index=meta.index)
    plot_df = pd.concat((plot_df, meta), axis=1, ignore_index=False)
    x_key = "umap_0"
    y_key = "umap_1"

    if tile is not False:
        tile_layout = np.array(tile_layout)
        tile_layout = np.atleast_2d(tile_layout)
        tile_groups = plot_df.groupby(tile)
        for group, group_data in tile_groups:
            group_index = group_data.index
            inds = np.where(tile_layout == group)
            plot_df.loc[group_index, x_key] += 2 * inds[1]
            plot_df.loc[group_index, y_key] += 2 * inds[0]

    results["plot_df"] = plot_df

    # TODO replace with generic color mapping logic
    # if cmap == "husl":
    #     colors = sns.color_palette("husl", plot_df[hue_key].nunique())
    # elif cmap == "glasbey":
    #     colors = cc.glasbey_light
    #     palette = dict(zip(plot_df[hue_key].unique(), colors))
    if node_palette is None and hue is not None:
        n_unique = meta[hue].nunique()
        if n_unique > 10:
            colors = cc.glasbey_light
        else:
            colors = sns.color_palette("Set2")
        node_palette = dict(zip(meta[hue].unique(), colors))

    if size == "degree":
        in_degree = np.asarray(adj.sum(axis=1))
        out_degree = np.asarray(adj.sum(axis=0))
        degree = np.squeeze(in_degree) + np.squeeze(out_degree)
        plot_df["degree"] = degree

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if verbose > 0:
        print("Scatterplotting nodes...")
    sns.scatterplot(
        data=plot_df,
        x=x_key,
        y=y_key,
        ax=ax,
        hue=hue,
        size=size,
        palette=node_palette,
        sizes=sizes,
        **scatterplot_kws,
    )
    node_paths = ax.get_children()
    ax.set(xlabel="", ylabel="", xticks=[], yticks=[])
    if not spines:
        for side in ["left", "right", "top", "bottom"]:
            ax.spines[side].set_visible(False)
    if not legend:
        ax.get_legend().remove()

    if group_convex_hull:
        groups = plot_df.groupby(group)
        for name, group_data in groups:
            points = group_data[[x_key, y_key]].values
            convex_hull = ConvexHull(points)
            ax.fill(
                points[convex_hull.vertices, 0],
                points[convex_hull.vertices, 1],
                "k",
                alpha=0.2,
                zorder=-1,
                linewidth=3,
                # linecolor="k",
                fill=False,
            )

    if verbose > 0:
        print("Collating edge data for plotting...")
    rows = []
    if networkx:
        for i, (pre, post) in enumerate(network.edges):
            rows.append({"pre": pre, "post": post, "edge_idx": i})
    else:
        pre_inds, post_inds = np.nonzero(adj)
        for i, (pre_ind, post_ind) in enumerate(zip(pre_inds, post_inds)):
            pre = index[pre_ind]
            post = index[post_ind]
            rows.append({"pre": pre, "post": post, "edge_idx": i})

    edgelist = pd.DataFrame(rows)

    if isinstance(subsample_edges, float):
        if verbose > 0:
            print("Subsampling edges...")
        n_edges = len(edgelist)
        n_show_edges = int(np.ceil(n_edges * subsample_edges))
        choice_inds = random_state.choice(n_edges, size=n_show_edges, replace=False)
        edgelist = edgelist.iloc[choice_inds]

    if verbose > 0:
        print("Mapping edge data for plotting...")
    if edge_hue == "prepost":
        # TODO this isn't working yet
        edgelist["prepost"] = list(zip(edgelist["pre"], edgelist["post"]))
    if edge_palette is None:
        edgelist["hue"] = edgelist[edge_hue].map(meta[hue])
    else:
        edgelist["hue"] = edgelist[edge_hue].map(edge_palette)

    pre_edgelist = edgelist.copy()
    post_edgelist = edgelist.copy()

    pre_edgelist["x"] = pre_edgelist["pre"].map(plot_df[x_key])
    pre_edgelist["y"] = pre_edgelist["pre"].map(plot_df[y_key])

    post_edgelist["x"] = post_edgelist["post"].map(plot_df[x_key])
    post_edgelist["y"] = post_edgelist["post"].map(plot_df[y_key])

    # plot_edgelist = pd.concat((pre_edgelist, post_edgelist), axis=0, ignore_index=True)

    # edge_palette = dict(zip(edgelist["edge_idx"], edgelist["hue"].map(palette)))

    pre_coords = list(zip(pre_edgelist["x"], pre_edgelist["y"]))
    post_coords = list(zip(post_edgelist["x"], post_edgelist["y"]))
    coords = list(zip(pre_coords, post_coords))
    edge_colors = edgelist["hue"].map(node_palette)

    if verbose > 0:
        print("Plotting edges...")
    lc = LineCollection(
        coords,
        colors=edge_colors,
        linewidths=edge_linewidth,
        alpha=edge_alpha,
        zorder=0,
    )
    ax.add_collection(lc)

    if hue_labels is not False:
        if verbose > 0:
            print("Labeling hue groups...")
        groupby = plot_df.groupby(hue)
        # centroids = groupby[[x_key, y_key]].mean()
        texts = []
        for label, group in groupby:
            points = group[[x_key, y_key]]
            pdists = pairwise_distances(points)
            medioid_ind = np.argmin(pdists.sum(axis=0))
            x, y = group.iloc[medioid_ind][[x_key, y_key]]
            if hue_labels == "radial":
                radial_scale = 1.02
                # radial_scale = 1.1
                vec = np.array((x, y))
                norm = np.linalg.norm(vec)
                vec *= radial_scale / norm
                x, y = vec
                if x < 0:
                    ha = "right"
                else:
                    ha = "left"
                if y < 0:
                    va = "top"
                else:
                    va = "bottom"
            if hue_labels == "medioid":
                ha = "center"
                va = "center"
            color = node_palette[label]
            text = ax.text(
                x,
                y,
                label,
                ha=ha,
                va=va,
                color=color,
                fontweight="bold",
                zorder=1000,
                fontsize=hue_label_fontsize,
            )
            text.set_bbox(dict(facecolor="white", alpha=0.7, linewidth=0, pad=0.5))
            texts.append(text)
        if adjust_labels:
            # arrowprops=dict(arrowstyle="->", color="black")
            if verbose > 0:
                print("Adjusting hue labels for overlaps...")
            # TODO doesn't seem like add_objects works here
            # ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
            # add_objects=node_paths
            adjust_text(
                texts,
                avoid_self=False,
                autoalign=False,
            )
    if return_results:
        return ax, results
    else:
        return ax
