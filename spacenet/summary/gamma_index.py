def gamma_index(spatial_network,is_planar=True):
    """
    The Gamma Index is a fundamental metric that measures a network's connectivity by comparing the actual number of edges to the maximum possible number of edges.
    A value close to 0 indicates poor connectivity, with relatively few connections.
    A value close to 1 indicates high connectivity, approaching the theoretical maximum for a planar network.
    
    Parameters
    ----------
    spatial_network : networkx.Graph    
        The spatial network for which to compute the gamma index.
        
    is_planar : bool, optional
        Indicates if the network is planar, by default True.
    
    Raises
    ------
    RuntimeError
        If the network name is not in the list of generated networks.
    
    Returns
    -------
    float
        The gamma index of the network.
    """
    
    numEdges=spatial_network.number_of_edges()
    numNodes=spatial_network.number_of_nodes()
    if is_planar==True:
        gamma=numEdges/(3*numNodes-6)
    else:
        gamma=(2*numEdges)/(numNodes*(numNodes-1))
    return gamma