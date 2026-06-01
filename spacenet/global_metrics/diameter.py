import networkx as nx

def diameter(spatial_network,edge_weight_name='Distance'):
    """
    
    The computes the diameter of the network, which is the maximum weighted shortest path between any two nodes in the network. 
    Also known as the maximum eccentricity.
    
    
    Parameters
    ----------
    spatial_network : networkx.Graph
        The spatial network for which to compute the diameter. 
    edge_weight_name : str, optional
        The name of the edge attribute in the graph that corresponds to the distance between nodes. Default is 'Distance'.
    
    Returns
    -------
    diameter_value : float
        The weighted diameter of the spatial network.
    
    
    Notes
    -----
    
    This is a wrapper to the NetworkX diameter function ``networkx.diameter()``. 
    
    """
    
    # this is a simple wrapper to networkx diameter
    diameter_value = nx.diameter(spatial_network,weight=edge_weight_name)
    
    return diameter_value