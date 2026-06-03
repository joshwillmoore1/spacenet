
from numbers import Number  
import networkx as nx
import collections
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl
import numpy as np



def draw_edges_three_d(
    G,
    pos,
    edgelist=None,
    width=1.0,
    edge_color="k",
    style="solid",
    alpha=None,
    edge_cmap=None,
    edge_vmin=None,
    edge_vmax=None,
    ax=None,
    label=None,
    nodelist=None,
    connectionstyle="arc3",
):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import matplotlib as mpl


    if ax is None:
        ax = plt.gca(projection='3d')

    if edgelist is None:
        edgelist = list(G.edges)

    if len(edgelist) == 0:
        return []

    if nodelist is None:
        nodelist = list(G.nodes())

    if edge_color is None:
        edge_color = "k"

    edge_pos = np.array([[pos[e[0]], pos[e[1]]] for e in edgelist])

    if (
        np.iterable(edge_color)
        and (len(edge_color) == len(edge_pos))
        and np.all([isinstance(c, Number) for c in edge_color])
    ):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, mpl.colors.Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        color_normal = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    edge_collection = Line3DCollection(
        edge_pos.reshape(-1, 2, 3),
        colors=edge_color,
        linewidths=width,
        linestyle=style,
        alpha=alpha,
    )
    edge_collection.set_cmap(edge_cmap)
    edge_collection.set_clim(edge_vmin, edge_vmax)
    edge_collection.set_zorder(1)
    edge_collection.set_label(label)
    ax.add_collection3d(edge_collection)

    return edge_collection