def get_node_node_distance(spatial_network,weight='Distance',sources=None):
    """
    Retrieves node-node distance information from a spatial network's distance cache for a specified weight attribute and an optional subset of source nodes.
    
    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network from which to retrieve node-node distance information.
    weight : str, optional
        The name of the edge attribute to use as the weight for retrieving node-node distance information. Default is 'Distance'.
    sources : array-like, optional
        A list or array of source node indices for which to retrieve shortest path distances to all other nodes in the graph. If None, distances for all sources in the cache will be retrieved. Default is None.
        
    Returns
    -------
    node_node_distances : dict
        A dictionary mapping each source node to a dictionary of shortest path distances to all other nodes in the graph. The inner dictionary maps target node indices to their corresponding shortest path distances from the source node.
    """
        
    this_distance_dictionary=spatial_network.distance_cache[weight]['distances']
    
    # get a subset of the distance dictionary for the sources of interest   
    if sources is not None:
        node_node_distances = {source: this_distance_dictionary[source] for source in sources if source in this_distance_dictionary}
    else:
        node_node_distances = this_distance_dictionary
    
    return node_node_distances