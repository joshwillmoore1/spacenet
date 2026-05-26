from numbers import Number  
import networkx as nx
import collections
import itertools
import matplotlib.pyplot as plt
import numpy as np
from spacenet.helpers import get_nodes_and_labels

def plot_spatial_network(spatial_network,node_label_name=None,nodes_to_plot=None,ax=None,edge_width=1,marker_size=10,add_node_cbar=True,edge_weight_name='Distance',edge_cmap='Greys_r',edge_vmin=None,edge_vmax=None,add_edge_cbar=False,scatter_kwargs={},figure_kwargs={}):
    """
    
    Plots a spatial network with nodes positioned according to the provided points.
    Nodes can be colored based on provided labels, and edges can be colored based on their weights. 

    Parameters
    ----------
    spatial_network : networkx.Graph
        The spatial network to be plotted. Edges should have a weight attribute corresponding to the distance between nodes.
    node_label_name : str, optional
        The name of the node attribute in the spatial network that corresponds to the labels for coloring nodes. If None, nodes will not be colored based on labels. Default is None.
    nodes_to_plot : list, tuple, or numpy array, optional
        A list, tuple, or numpy array of node indices to plot. If None, all nodes in the spatial network will be plotted. Default is None.
    ax : matplotlib.axes.Axes, optional 
        An existing matplotlib Axes object to plot on. If None, a new figure and axes will be created. Default is None.
    edge_width : float, optional
        The width of the edges in the plot. Default is 1.
    marker_size : float, optional
        The size of the node markers in the plot. Default is 10.
    add_node_cbar : bool, optional
        Whether to add a legend for the node labels. Default is False.
    edge_weight_name : str, optional
        The name of the edge attribute in the spatial network that corresponds to the distance between nodes. Default is 'Distance'.
    edge_cmap : str or matplotlib.colors.Colormap, optional
        The colormap to use for coloring edges based on their weights. Default is 'Greys_r'.
    edge_vmin : float, optional
        The minimum value for edge weight coloring. If None, the minimum edge weight will be used. Default is None.
    edge_vmax : float, optional
        The maximum value for edge weight coloring. If None, the maximum edge weight will be used. Default is None.
    add_edge_cbar : bool, optional
        Whether to add a colorbar for the edge weights. Default is False.
     scatter_kwargs : dict, optional
        Additional keyword arguments to pass to the scatter function when plotting nodes. Default is an empty dictionary.
    figure_kwargs : dict, optional
        Additional keyword arguments to pass to plt.figure() when creating a new figure. Default is an empty dictionary.

    Returns
    -------
    fig : matplotlib.figure.Figure  
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.

                            
    """
    
    
    # get the nodes we are interested in plotting based on the provided nodes_to_plot argument
    if nodes_to_plot is None:
        node_indices = list(spatial_network.nodes())    
    elif isinstance(nodes_to_plot, (list, tuple, np.ndarray)):
        node_indices = nodes_to_plot
    else:
        raise ValueError("nodes_to_plot must be None or a list, tuple, or numpy array of node indices.")
    
    # get the node locations from the spatial network
    node_indices_all = list(spatial_network.nodes())
    points_all = np.array([spatial_network.nodes[node]['position'] for node in node_indices_all])
    pos_all = {id: points_all[i] for i,id in enumerate(node_indices_all)}
    
    # only those nodes we are interested in plotting
    points = np.array([spatial_network.nodes[node]['position'] for node in node_indices])
    
    # get the node labels from the spatial network
    if node_label_name is not None:
        q_nodes,q_labels = get_nodes_and_labels(spatial_network,node_label_name)
        label_dict = {node: label for node, label in zip(q_nodes, q_labels)}
        labels = np.array([label_dict[node] for node in node_indices])
        
        # if labels are floats, then we will treat them as continuous values 
        if isinstance(labels[0], (float)):
            continuous_labels = True
        else:
            continuous_labels = False
        
    else:
        labels = None   
        
    if ax is None:
        
        if points.shape[1]==2:
            figure_kwargs={'figsize':(10,8)}|figure_kwargs
            fig = plt.gcf()
            ax = plt.gca()
    
        elif points.shape[1]==3:
            from mpl_toolkits.mplot3d import Axes3D
            figure_kwargs={'figsize':(10,8)}|figure_kwargs
            fig = plt.figure(**figure_kwargs)
            ax = fig.add_subplot(111, projection='3d')
        else:
            raise ValueError('Points must be of shape (n,2) or (n,3). Only 2D and 3D plotting supported.')
        
    if labels is None:
        if points.shape[1]==2:
            ax.scatter(points[:, 0], points[:, 1],color='grey', s=marker_size,zorder=100,**scatter_kwargs)
        elif points.shape[1]==3:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],color='grey', s=marker_size,zorder=100,**scatter_kwargs)

    else:
        
        if continuous_labels:
            
            if points.shape[1]==2:
                this_scat = ax.scatter(points[:, 0], points[:, 1], c=labels, s=marker_size,zorder=100,**scatter_kwargs)
            elif points.shape[1]==3:
                this_scat = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, s=marker_size,zorder=100,**scatter_kwargs)
            if add_node_cbar:   
                cbar = plt.colorbar(this_scat, ax=ax)
                cbar.set_label(node_label_name)
            
        else:
            unique_labels = np.unique(labels)
            
            if points.shape[1]==2:
                for i in unique_labels:
                    ax.scatter(points[labels==i, 0], points[labels==i, 1], label=str(i), s=marker_size,zorder=100,**scatter_kwargs)
            elif points.shape[1]==3:
                for i in unique_labels:
                    ax.scatter(points[labels==i, 0], points[labels==i, 1], points[labels==i, 2], label=str(i), s=marker_size,zorder=100,**scatter_kwargs)
            if add_node_cbar:
                ax.legend()
    
    if nx.is_weighted(spatial_network, edge=None, weight=edge_weight_name):
        edge_cmap=plt.get_cmap(edge_cmap)
        _, weights = zip(*nx.get_edge_attributes(spatial_network,edge_weight_name).items())
        
        if points.shape[1]==2:
        
            edges=__draw_edges_2D(spatial_network, 
                                pos_all,
                                edge_color = weights,
                                edge_cmap = edge_cmap, 
                                width=edge_width,
                                edge_vmin=edge_vmin,
                                edge_vmax=edge_vmax,
                                ax=ax)
        elif points.shape[1]==3:
            edges=__draw_edges_3D(spatial_network, 
                                pos_all,
                                edge_color = weights,
                                edge_cmap = edge_cmap, 
                                width=edge_width,
                                edge_vmin=edge_vmin,
                                edge_vmax=edge_vmax,
                                ax=ax)
        if add_edge_cbar:
            edge_cbar=plt.gcf().colorbar(edges,ax=ax)
            edge_cbar.set_label(edge_weight_name)
    else:
        
        if points.shape[1]==2:
            edges=__draw_edges_2D(spatial_network, 
                                pos_all,
                                edge_color = [0.2,0.2,0.2,1], 
                                width=edge_width ,
                                edge_vmin=edge_vmin,
                                edge_vmax=edge_vmax,
                                ax=ax)
        elif points.shape[1]==3:
            edges=__draw_edges_3D(spatial_network, 
                                pos_all,
                                edge_color = [0.2,0.2,0.2,1], 
                                width=edge_width ,
                                edge_vmin=edge_vmin,
                                edge_vmax=edge_vmax,
                                ax=ax)
    
    # Set aspect ratio to equal
    ax.set_aspect('equal')
    ax.set_axis_on()
    return plt.gcf(),ax

def __draw_edges_2D(
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

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np


    # The default behavior is to use LineCollection to draw edges for
    # undirected graphs (for performance reasons) and use FancyArrowPatches
    # for directed graphs.
    # The `arrows` keyword can be used to override the default behavior
   

    if isinstance(connectionstyle, str):
        connectionstyle = [connectionstyle]
    elif np.iterable(connectionstyle):
        connectionstyle = list(connectionstyle)
    else:
        msg = "draw_networkx_edges arg `connectionstyle` must be str or iterable"
        raise nx.NetworkXError(msg)

    # Some kwargs only apply to FancyArrowPatches. Warn users when they use
    # non-default values for these kwargs when LineCollection is being used
    # instead of silently ignoring the specified option



    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges)  # (u, v, k) for multigraph (u, v) otherwise

    if len(edgelist):
        if G.is_multigraph():
            key_count = collections.defaultdict(lambda: itertools.count(0))
            edge_indices = [next(key_count[tuple(e[:2])]) for e in edgelist]
        else:
            edge_indices = [0] * len(edgelist)
    else:  # no edges!
        return []

    if nodelist is None:
        nodelist = list(G.nodes())

    # FancyArrowPatch handles color=None different from LineCollection
    if edge_color is None:
        edge_color = "k"

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    # Check if edge_color is an array of floats and map to edge_cmap.
    # This is the only case handled differently from matplotlib
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

    # Draw the edges
    edge_collection = mpl.collections.LineCollection(
        edge_pos,
        colors=edge_color,
        linewidths=width,
        antialiaseds=(1,),
        linestyle=style,
        alpha=alpha,
    )
    edge_collection.set_cmap(edge_cmap)
    edge_collection.set_clim(edge_vmin, edge_vmax)
    edge_collection.set_zorder(1)  # edges go behind nodes
    edge_collection.set_label(label)
    ax.add_collection(edge_collection)
    edge_viz_obj = edge_collection

    return edge_viz_obj

def __draw_edges_3D(
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