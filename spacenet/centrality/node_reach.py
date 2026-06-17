import numpy as np
from spacenet.helpers import node_node_distance
from spacenet.utils import add_node_labels

def node_reach(spatial_network,nodes=None, max_distance=np.inf, edge_weight_name='Distance',add_as_node_label=False,node_label_name='node_reach'):
    """
    
    Computes the node reach centrality for the nodes in the spatial network.
    This is the proportion of nodes within the network that are reachable from a given node within a certain distance. 
    
    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network for which to calculate the node reach centrality.
    nodes : list or np.ndarray, optional
        The set of nodes to compute the centrality. Default is 'None', computing for all nodes.
    max_distance : float, optional
        The maximum distance to consider when computing the centrality. Only pairs of nodes that are within this distance of each other will be included in the computation. Default is np.inf (no limit).
    edge_weight_name : str, optional
        The name of the edge attribute in the graph that corresponds to the distance between nodes. Default is 'Distance'.
    add_as_node_label : bool, optional
        Whether to add the computed centrality values as a node attribute in the spatial network. Default is False.
    node_label_name : str, optional
        The name of the node attribute to which to add the computed centrality values if add_as_node_label is True. Default is 'node_reach'.
    
    
    Returns
    -------
    
    reach_values : np.ndarray
        An array of node reach centrality values for the specified nodes in the spatial network. The order of the values corresponds to the order of the nodes in the 'nodes' parameter.
    nodes : np.ndarray
        An array of the node ids for which the node reach centrality values were computed. The order of the nodes corresponds to the order of the values in the 'reach_values' array.
    
    
    Examples
    --------
    You can compute the node reach centrality of nodes in a spatial network using the `node_reach` function. Below is an example of how to use this function to compute the node reach centrality for all nodes in a spatial network generated from a set of points.
    
    .. code-block:: python
    
        import spacenet as sn

        # Load the spiral dataset and extract the 'x' and 'y' columns as points
        spiral_data = sn.datasets.load_dataset('spiral')
        points = spiral_data[['x', 'y']].to_numpy()

        # generate a spatial network
        G = sn.utils.spatial_network_from_points(points,max_edge_distance=50)

        # compute the node reach for each node in the spatial network and add it as a node label
        node_reach_vals,node_ids=sn.centrality.node_reach(G,max_distance=200,add_as_node_label=True,node_label_name='node_reach')

        # plot the spatial network and with the node label 'node_reach' to visualise the node reach centrality values
        sn.utils.plot_spatial_network(G,node_label_name='node_reach')
    
    """
    
    # if no nodes give run on all
    if nodes is None:
        nodes = np.array(list(spatial_network.nodes))
    elif isinstance(nodes,(list,np.ndarray)):
        nodes = np.array(nodes)
        
        # validate nodes
        all_nodes = np.array(list(spatial_network.nodes))
        valid_nodes_check = np.isin(nodes,all_nodes,assume_unique=True)
        
        if np.sum(valid_nodes_check)!=len(nodes):
            raise ValueError(f'Nodes passed that are not nodes of this spatial network. These nodes are {nodes[~valid_nodes_check]}')
        
    else:
        raise ValueError(f'nodes not of the correct type. Acceptable types are list or array of node ids but given here f{type(nodes)}.')
    
    # get the distances
    node_distances = node_node_distance(spatial_network,sources=nodes,weight=edge_weight_name,limit=max_distance)
    
    # compute closeness for each node using the Wasserman and Faust formula 
    reach_values = np.zeros(len(nodes))
    
    
    for i, node in enumerate(nodes):
        these_distances = node_distances[node]
        in_reach_count = 0
        for target in these_distances:
            if target != node:
                this_dist_val = these_distances[target]
                if this_dist_val <= max_distance:
                    in_reach_count += 1
        reach_values[i] = in_reach_count 
        
        
    # scaling for Wasserman and Faust formula 
    reach_values = reach_values / len(list(spatial_network.nodes))
    
    if add_as_node_label:
        add_node_labels(spatial_network,reach_values,node_label_name=node_label_name,nodes=nodes)
    
    return reach_values, nodes