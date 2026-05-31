def beta_index(spatial_network):
    """
    The beta index is the ratio of edges to nodes in a spatial network.
    It is a measure of the connectivity of the network, with higher values indicating a more connected network.

    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network for which to calculate the beta index.

    Returns
    -------
    beta_index : float
        The beta index of the spatial network.
    
    """
    
    num_edges = spatial_network.number_of_edges()
    num_nodes = spatial_network.number_of_nodes()
    
    if num_nodes == 0:
        return 0
    
    beta_value = num_edges / num_nodes
    
    return beta_value