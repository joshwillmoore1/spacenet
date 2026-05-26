def alpha_index(spatial_network):
    """    
    
    The alpha index, also knowned as the meshedness coefficient, is a metric used to evaluate and measure the connectivity of a network. 
    It is calculated as the ratio between the actual number of fundamental circuits (cycles) and the maximum possible number of circuits in that network.
    A tree-like network has alpha ~0 and a max planar network has alpha ~1.
    
    Parameters
    ----------
    spatial_network : networkx.Graph    
        The spatial network for which to compute the alpha index.
    
    Returns
    -------
    alpha : float
        The alpha index of the network.

    """
    
    numEdges=spatial_network.number_of_edges()
    numNodes=spatial_network.number_of_nodes()
    alpha=(numEdges-numNodes+1)/(2*numNodes-5)
    
    return alpha