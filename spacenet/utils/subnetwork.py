import networkx as nx
import numpy as np
from copy import deepcopy
from spacenet.helpers.remove_nodes_from_distance_cache import remove_nodes_from_distance_cache


def subnetwork(spatial_network, nodes=None):
    """
    
    Creates a subnetwork of a spatial network by including only the specified nodes and the edges between them. 
    If nodes is None, the original spatial network is returned.
    
    
    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network from which to create the subnetwork.
    nodes : array-like, optional
        A list or array of node indices to include in the subnetwork. If None, the original spatial network is returned. Default is None.
        
    Returns
    -------
    subnetwork : NetworkX graph
        A subnetwork of the original spatial network that includes only the specified nodes and the edges between them. If nodes is None, the original spatial network is returned.
    
    """
    
    # make a copy of the spatial network
    subnetwork = deepcopy(spatial_network)
    original_nodes = set(spatial_network.nodes())
    
    if nodes is not None:
        if isinstance(nodes,(list,np.ndarray)):
            nodes = set(nodes)
        else:
            raise ValueError("nodes must be a list or array of node indices.")
        
        # get the nodes to remove
        nodes_to_remove = original_nodes - nodes
        
        # remove the nodes from the subnetwork
        subnetwork.remove_nodes_from(list(nodes_to_remove))
        
        # remove the corresponding entries from the distance cache
        remove_nodes_from_distance_cache(subnetwork, list(nodes_to_remove))
        
    return subnetwork
    
    