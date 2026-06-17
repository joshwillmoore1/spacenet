from scipy.sparse import csr_array
import numpy as np
import networkx as nx

def spatial_network_from_adjacency(points, adjacency_matrix, inverse_distance_function=None,node_ids=None):
    """
    
    Generates a spatial network from a given adjacency matrix and node coordinates. 
    The function creates a NetworkX graph object, adds nodes and edges, and computes inverse distances for the edges based on the provided or default inverse distance function.
    Edge weights for 'Distance' and 'Inverse Distance' will be added to the network. 
    Node indices correspond to the indices of the input points array.
    Nodes will have a 'position' attribute corresponding to their spatial coordinates from the input points array.
    
    Parameters
    ----------
    points : numpy.ndarray
        An array of shape (n, 2) representing the coordinates of the nodes in the network. Points should be ordered in ascending order of node ids in the adjacency matrix.
    adjacency_matrix : np.ndarray or csr_array
        A weighted adjacency array where non-zero entries represent edges in the spatial network. The values will be considered as the distances between nodes.
    inverse_distance_function : callable, optional
        A function that takes a distance as an argument and returns an inverse distance value.
    node_ids : array-like, optional 
        An array of custom node identifiers corresponding to the input points. If None, enumerated node indices will be used as identifiers. Default is None.
   
    Returns
    -------
    
    G : networkx.Graph
        A NetworkX graph object representing the spatial network, with nodes and edges added.
    """
    
    
    # adjacency_matrix should be a square numpy array or sparse scipy sparse array but with values representing distances
    if not isinstance(adjacency_matrix, (np.ndarray,csr_array)):
        print(type(adjacency_matrix))
        raise ValueError("adjacency_matrix should be a square numpy array or scipy sparse matrix with values representing distances")
    
    # Convert the adjacency matrix to a sparse CSR matrix if it's not already
    if not isinstance(adjacency_matrix, csr_array):
        adjacency_matrix = csr_array(adjacency_matrix)
    
    
    #check if the adjacency matrix is square
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("adjacency_matrix should be square")

    # check there is enough points for the number of nodes in the adjacency matrix
    if adjacency_matrix.shape[0] != len(points):
        raise ValueError("Number of points does not match number of nodes in the adjacency matrix.")
    
    max_edge_distance = adjacency_matrix.max()
    
    if inverse_distance_function is not None:
        #check if the inverse distance function is callable
        if callable(inverse_distance_function):
            this_inverse_distance_function=inverse_distance_function
        else:
            raise ValueError(f"Parameter inverse_distance_function is not callable: please pass a function that takes a distance as an argument and returns a value") 
    else:
        this_inverse_distance_function=lambda x: 1/(1+(4/max_edge_distance)*x)   
    
    
    if node_ids is not None:
        if len(node_ids) != adjacency_matrix.shape[0]:
            raise ValueError("Length of node_ids does not match number of nodes in the adjacency matrix.")
        all_nodes = np.array(node_ids)
    else:
        all_nodes = np.arange(adjacency_matrix.shape[0])
    
    
    edgelist = []
    indices = zip(*adjacency_matrix.nonzero())
    for i, j in indices:
        edgelist.append((all_nodes[i], all_nodes[j], adjacency_matrix[i, j]))
    edgelist = np.array(edgelist)
    
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    
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
    
    