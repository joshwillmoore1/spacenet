import numpy as np

def get_edge_weights(network,weight='Distance'):  
    """
    
    Extracts edge weights from a spatial network for a specified weight attribute and returns them as a numpy array.
    
    Parameters
    ----------
    network : networkx.Graph
        The spatial network from which to extract edge weights. Edges should have a weight attribute corresponding to the distance between nodes.
        
    weight : str, optional
        The name of the edge attribute in the network that corresponds to the distance between nodes. Default is 'Distance'.
    
    Returns
    -------
    edge_weights : numpy.ndarray
        An array of edge weights corresponding to the specified weight attribute for each edge in the network.
    
    """
    #if weight not in network.edges(data=True)[0][2]:
    #    raise ValueError(f"Weight '{weight}' is not a valid edge attribute.")
    
    edge_weights = np.asarray([data[weight] for _, _, data in network.edges(data=True)])
    
    return edge_weights