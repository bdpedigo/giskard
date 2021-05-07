# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspologic.utils import import_graph
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import normalize

import colorcet as cc


def graphplot(
    data,
    meta=None,
    input_graph=True,
    embedding_algorithm="ase",
    n_components=8,
    n_neighbors=15,
    min_dist=0.8,
    metric="cosine",
    hue=None,
    size="degree",
    palette=None,
    ax=None,
    figsize=(10, 10),
    sizes=(5, 10),
    legend=False,
    edge_hue="pre",
    edge_linewidth=0.2,
    edge_alpha=0.2,
    spines=False,
    subsample_edges=False,
    verbose=False,
    random_state=None,
    network_order=1,
    normalize_power=False,
    supervised_weight=False,
    embed_kws={},
    umap_kws={},
    scatterplot_kws={},
):
    if random_state is None:
        random_state = np.random.default_rng()
    if input_graph:
        # if we are given a graph, do an initial embedding
        adj = import_graph(data, to_csr=True)

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
            if normalize_power:
                adj_normed = normalize(adj, axis=1)
                embedding = embedder.fit_transform(adj_normed @ adj_normed)
            else:
                embedding = embedder.fit_transform(adj @ adj)
        elif network_order == 1:
            embedding = embedder.fit_transform(adj)
    else:
        raise NotImplementedError("Currently not supporting inputs of embeddings.")

    # if input is networkx, extract node metadata into a data frame
    if isinstance(data, (nx.Graph, nx.DiGraph)):
        networkx = True
        index = list(sorted(data.nodes()))
        meta = pd.DataFrame(index=index)
        for attr in [hue, size]:
            if attr is not None:
                attr_map = nx.get_node_attributes(data, attr)
                meta[attr] = meta.index.map(attr_map)
    else:
        networkx = False
        index = meta.index

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
        y = meta[hue].values
        _, y = np.unique(y, return_inverse=True)
    else:
        y = None
    umap_embedding = umapper.fit_transform(embedding, y=y)

    # TODO
    # mids = (umap_embedding.max(axis=0) + umap_embedding.min(axis=0)) / 2
    # umap_embedding -= mids
    # max_length = np.linalg.norm(umap_embedding, axis=1).max()
    # umap_embedding /= max_length

    # add the UMAP embedding into the dataframe for plotting
    columns = [f"umap_{i}" for i in range(umap_embedding.shape[1])]
    plot_df = pd.DataFrame(data=umap_embedding, columns=columns, index=meta.index)
    plot_df = pd.concat((plot_df, meta), axis=1, ignore_index=False)

    x_key = "umap_0"
    y_key = "umap_1"

    # TODO color mapping logic
    # if cmap == "husl":
    #     colors = sns.color_palette("husl", plot_df[hue_key].nunique())
    # elif cmap == "glasbey":
    #     colors = cc.glasbey_light
    #     palette = dict(zip(plot_df[hue_key].unique(), colors))
    if palette is None and hue is not None:
        n_unique = meta[hue].nunique()
        if n_unique > 10:
            colors = cc.glasbey_light
        else:
            colors = sns.color_palette("deep")
        palette = dict(zip(meta[hue].unique(), colors))

    if size == "degree":
        in_degree = np.asarray(adj.sum(axis=1))
        out_degree = np.asarray(adj.sum(axis=0))
        degree = in_degree + out_degree
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
        palette=palette,
        sizes=sizes,
        **scatterplot_kws,
    )
    ax.set(xlabel="", ylabel="", xticks=[], yticks=[])
    if not spines:
        for side in ["left", "right", "top", "bottom"]:
            ax.spines[side].set_visible(False)
    if not legend:
        ax.get_legend().remove()

    if verbose > 0:
        print("Collating edge data for plotting...")
    rows = []
    if networkx:
        for i, (pre, post) in enumerate(data.edges):
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
    edgelist["hue"] = edgelist[edge_hue].map(meta[hue])

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
    edge_colors = edgelist["hue"].map(palette)

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
    return ax
