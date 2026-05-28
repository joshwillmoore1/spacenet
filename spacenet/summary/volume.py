from spacenet.helpers.get_edge_weights import get_edge_weights

def volume(spatial_network, edge_weight_name='Distance'):
    """
    Computes the total volume of a spatial network, which is defined as the sum of the edge weights specified by edge_weight_name.
    The default edge weight is 'Distance', which represents the physical distance between nodes in the spatial network.
    The function checks that the specified edge weight exists in the graph and then sums up the values of that edge weight for all edges in the graph to compute the total volume.
    
    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network for which to compute the volume.
    edge_weight_name : str, optional
        The name of the edge attribute to use as the weight for computing the volume. Default is 'Distance'.
    
    Returns
    -------
    total_volume : float
        The total volume of the spatial network, which is the sum of the edge weights specified by edge_weight_name.
    
    """
    # check that edge_weight_name is a weight in the graph
    if edge_weight_name not in spatial_network.edges[next(iter(spatial_network.edges))]:
        raise ValueError(f"Edge weight '{edge_weight_name}' not found in the graph. Please provide a valid edge weight name.")
    
    # get the volume of the network, which is the sum of the edge weights
    distances = get_edge_weights(spatial_network, weight=edge_weight_name)
    total_volume = sum(distances)
    
    return total_volume