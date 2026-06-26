import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from spacenet.helpers import get_nodes_and_labels
from spacenet.utils.helpers import draw_edges_two_d, draw_edges_three_d
from matplotlib.colors import ListedColormap, BoundaryNorm


def plot_spatial_network(spatial_network,node_label_name=None,nodes_to_plot=None,ax=None,edge_width=1,marker_size=10,add_node_cbar=True,edge_weight_name=None,edge_cmap='Greys_r',edge_vmin=None,edge_vmax=None,add_edge_cbar=False,scatter_kwargs={},figure_kwargs={}):
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
        The name of the edge attribute in the spatial network that corresponds to the distance between nodes. Default is None.
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
        
    Examples
    --------
    You can use the `plot_spatial_network` function to visualize a spatial network with nodes colored by their labels and edges colored by their weights. Below is a few examples of how to use this function to plot a spatial network generated from a set of points.
    The basic usage of the function is as follows:
    
    .. code-block:: python

        import spacenet as sn

        # get data from the Spiral dataset
        sprial_df = sn.datasets.load_dataset('spiral')
        points = sprial_df[['x','y']].values

        # generate a spatial network using the delaunay method
        G = sn.utils.spatial_network_from_points(points,network_type='delaunay',max_edge_distance=75)

        # plot the spatial network
        sn.utils.plot_spatial_network(G)
        
    The same functionality works for 3D spatial networks as well. 
    
    .. code-block:: python
    
        import spacenet as sn

        # get data from the Cylinder dataset
        sprial_df = sn.datasets.load_dataset('cylinder')
        points = sprial_df[['x','y','z']].values

        # generate a spatial network using the delaunay method
        G = sn.utils.spatial_network_from_points(points,network_type='delaunay',max_edge_distance=30)

        # plot the spatial network
        sn.utils.plot_spatial_network(G)
        
    Node labels can also be shown on these plots by providing the node_label_name argument, the visualisation will infer whether this is a categorical or continuous label. For example, if we have a categorical label called 'Marker (categorical)' in the spatial network, we can plot the spatial network with nodes colored by this label as follows:
    
    .. code-block:: python

        import spacenet as sn

        # get data from the Spiral dataset
        sprial_df = sn.datasets.load_dataset('spiral')
        points = sprial_df[['x','y']].values
        categorical_labels = sprial_df['Marker (categorical)'].values

        # generate a spatial network using the delaunay method and add labels
        G = sn.utils.spatial_network_from_points(points,network_type='delaunay',max_edge_distance=75)
        sn.utils.add_node_labels(G,categorical_labels,node_label_name='Marker (categorical)')

        # plot the spatial network with nodes coloured by their categorical label
        sn.utils.plot_spatial_network(G,node_label_name='Marker (categorical)',add_node_cbar=True)
                            

    If we have a continuous label called 'Marker (continuous)' in the spatial network, we can plot the spatial network with nodes colored by this label as follows:
    
    .. code-block:: python  
    
        import spacenet as sn

        # get data from the Cylinder dataset
        sprial_df = sn.datasets.load_dataset('cylinder')
        points = sprial_df[['x','y','z']].values
        categorical_labels = sprial_df['Marker (continuous)'].values

        # generate a spatial network using the delaunay method and add labels
        G = sn.utils.spatial_network_from_points(points,network_type='delaunay',max_edge_distance=35)
        sn.utils.add_node_labels(G,categorical_labels,node_label_name='Marker (continuous)')

        # plot the spatial network with nodes coloured by their continuous label
        sn.utils.plot_spatial_network(G,node_label_name='Marker (continuous)',add_node_cbar=True)
        
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
        
        
        # this assumes all nodes will have these values 
        labels = [label_dict[node] if node in label_dict else 'null' for node in node_indices]
        
        # if labels are floats, then we will treat them as continuous values 
        
        not_null_labels = [label for label in labels if label != 'null']
        if len(not_null_labels) == 0:
            raise ValueError(f"No valid labels found for node_label_name '{node_label_name}'.")
        
        if isinstance(not_null_labels[0], (str,int,np.integer)):
            continuous_labels = False
        else:
            continuous_labels = True
                    
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
        
    if nx.is_weighted(spatial_network, edge=None, weight=edge_weight_name):
        edge_cmap=plt.get_cmap(edge_cmap)
        _, weights = zip(*nx.get_edge_attributes(spatial_network,edge_weight_name).items())
        
        if points.shape[1]==2:
        
            edges=draw_edges_two_d(spatial_network, 
                                pos_all,
                                edge_color = weights,
                                edge_cmap = edge_cmap, 
                                width=edge_width,
                                edge_vmin=edge_vmin,
                                edge_vmax=edge_vmax,
                                ax=ax)
        elif points.shape[1]==3:
            edges=draw_edges_three_d(spatial_network, 
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
            edges=draw_edges_two_d(spatial_network, 
                                pos_all,
                                edge_color = [0.2,0.2,0.2,1], 
                                width=edge_width ,
                                edge_vmin=edge_vmin,
                                edge_vmax=edge_vmax,
                                ax=ax)
        elif points.shape[1]==3:
            edges=draw_edges_three_d(spatial_network, 
                                pos_all,
                                edge_color = [0.2,0.2,0.2,1], 
                                width=edge_width ,
                                edge_vmin=edge_vmin,
                                edge_vmax=edge_vmax,
                                ax=ax)
        else: 
            raise ValueError('Node position must be of shape (n,2) or (n,3). Only 2D and 3D plotting supported.')
        
    if labels is None:
        if points.shape[1]==2:
            ax.scatter(points[:, 0], points[:, 1],color='grey', s=marker_size,zorder=100,**scatter_kwargs)
        elif points.shape[1]==3:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],color='grey', s=marker_size,zorder=100,**scatter_kwargs)

    else:
        
        if continuous_labels:
            
            if 'vmax' not in scatter_kwargs:
                max_not_null_label = np.max(not_null_labels)
            else:
                max_not_null_label = scatter_kwargs['vmax']
                scatter_kwargs.pop('vmax')
            if 'vmin' not in scatter_kwargs:
                min_not_null_label = np.min(not_null_labels)
            else:
                min_not_null_label = scatter_kwargs['vmin']
                scatter_kwargs.pop('vmin')
                
            null_value = min_not_null_label-1
            labels = np.array([label if label != 'null' else null_value for label in labels])
            
            # make a colormap that maps the null value to grey and the rest of the values to the provided colormap
            if 'cmap' not in scatter_kwargs:
                colormap = plt.get_cmap('viridis')
            else:
                colormap = plt.get_cmap(scatter_kwargs['cmap'])
            
            norm = plt.Normalize(vmin=min_not_null_label, vmax=max_not_null_label)
            scatter_kwargs.pop('norm', None)
            scatter_kwargs.pop('cmap', None)
            colormap.set_under('grey')  # Set color for values below vmin (null values)
            
            if points.shape[1]==2:
                this_scat = ax.scatter(points[:, 0], points[:, 1], c=labels, s=marker_size,zorder=100,norm=norm,cmap=colormap,**scatter_kwargs)
            elif points.shape[1]==3:
                this_scat = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, s=marker_size,zorder=100,cmap=colormap,norm=norm,**scatter_kwargs)
            if add_node_cbar:   
                cbar = plt.colorbar(this_scat, ax=ax)
                cbar.set_label(node_label_name)
            
        else:
            unique_labels = np.unique(not_null_labels)
            unique_labels_all = list(unique_labels) +  ['null']
            if 'cmap' not in scatter_kwargs:
                colormap = plt.get_cmap('tab10')
            else:
                colormap = plt.get_cmap(scatter_kwargs['cmap'])
                
            cmap_dict = {label: colormap(i) for i, label in enumerate(unique_labels)}
            cmap_dict['null'] = (0.5, 0.5, 0.5, 1)
            
            # Build discrete colors for the categories
            category_colors = list(cmap_dict.values())
            cmap = ListedColormap(category_colors)
            
            # Map labels -> integer codes
            label_to_idx = {label: i for i, label in enumerate(unique_labels_all)}
            label_idx = np.array([label_to_idx[label] for label in labels])

            # BoundaryNorm makes the colorbar categorical rather than continuous
            norm = BoundaryNorm(np.arange(len(unique_labels) + 1) - 0.5, len(unique_labels))

            # Avoid passing a conflicting cmap in scatter_kwargs
            scatter_kwargs = dict(scatter_kwargs)
            scatter_kwargs.pop('cmap', None)
            scatter_kwargs.pop('norm', None)
            
            if points.shape[1]==2:
                this_scat=ax.scatter(points[:, 0], points[:, 1], c=label_idx, s=marker_size,zorder=100,cmap=cmap, norm=norm,**scatter_kwargs)
            elif points.shape[1]==3:
                this_scat=ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=label_idx, s=marker_size,zorder=100,cmap=cmap, norm=norm,**scatter_kwargs)
            if add_node_cbar:
                cbar = plt.colorbar(this_scat, ax=ax, ticks=np.arange(len(unique_labels)))
                cbar.ax.set_yticklabels(unique_labels)
                cbar.set_label(node_label_name)
    

    # Set aspect ratio to equal
    ax.set_aspect('equal')
    ax.set_axis_on()
    
    return plt.gcf(),ax
