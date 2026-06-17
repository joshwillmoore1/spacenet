import numpy as np
import networkx as nx

def spatial_network_from_edgelist(points, edgelist, inverse_distance_function=None):
    """
    
    Generates a spatial network from a given edgelist and node coordinates. 
    The function creates a NetworkX graph object, adds nodes and edges, and computes inverse distances for the edges based on the provided or default inverse distance function.
    Edge weights for 'Distance' and 'Inverse Distance' will be added to the network. 
    Node indices correspond to the indices of the input points array.
    Nodes will have a 'position' attribute corresponding to their spatial coordinates from the input points array.
    
    Parameters
    ----------
    points : numpy.ndarray
        An array of shape (n, 2) representing the coordinates of the nodes in the network. Points should be ordered in ascending order of node ids in the edgelist.
    edgelist : list of tuples
        A list of tuples (node1, node2, weight) representing the edges of the network.
    inverse_distance_function : callable, optional
        A function that takes a distance as an argument and returns an inverse distance value.
        
    Returns
    -------
    G : networkx.Graph
        A NetworkX graph object representing the spatial network, with nodes and edges added.
    """
    
    # edgelist should be a list of tuples (node1, node2, weight)
    if not isinstance(edgelist, (list, np.ndarray)):
        raise ValueError("edgelist should be a list or numpy array of tuples (node1, node2, weight)")
    
    edgelist  = np.array(edgelist)
    max_edge_distance = np.max(edgelist[:,2])
    
    if inverse_distance_function is not None:
        #check if the inverse distance function is callable
        if callable(inverse_distance_function):
            this_inverse_distance_function=inverse_distance_function
        else:
            raise ValueError(f"Parameter inverse_distance_function is not callable: please pass a function that takes a distance as an argument and returns a value") 
    else:
        this_inverse_distance_function=lambda x: 1/(1+(4/max_edge_distance)*x)   
        
    # edgelist should be a list of tuples (node1, node2, weight)
    G = nx.Graph()
    all_nodes = set()
    for edge in edgelist:
        all_nodes.update(edge[:2])
    
    all_nodes = np.sort(np.array(list(all_nodes),dtype=int))    
    G.add_nodes_from(all_nodes)
    
    # assumsing that points will ordered in ascending order of node ids in the edgelist
    if len(all_nodes) != len(points):
        raise ValueError("Number of points does not match number of nodes in the edgelist.")
        
    # Prepare edge lists with distances and inverse distances
    final_edge_list = edgelist
    final_edge_list_inv = np.hstack([edgelist[:,:2], this_inverse_distance_function(edgelist[:,2])[:, None]])
    
    # dump the edges into the network
    G.add_weighted_edges_from(final_edge_list, weight='Distance')
    G.add_weighted_edges_from(final_edge_list_inv, weight='Inverse Distance')
    
    # add positions as node attributes
    for i,id in enumerate(all_nodes):
        G.nodes[id]['position'] = points[i]
        
    # add an empty distance cache
    G.distance_cache = {}
    
    return G