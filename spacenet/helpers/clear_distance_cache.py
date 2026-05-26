def clear_distance_cache(spatial_network):
    
    """
    Clear the distance cache of a spatial network. This is useful if you have modified the network (e.g. added or removed edges) and want to ensure that distance calculations are updated accordingly.
    
    Parameters
    ----------
    spatial_network : networkx.Graph
        The spatial network for which to clear the distance cache.
        
    Returns
    -------
    None
    
    """
    
    if hasattr(spatial_network, 'distance_cache'):
        spatial_network.distance_cache = {}