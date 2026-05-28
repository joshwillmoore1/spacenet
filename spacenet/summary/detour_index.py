import numpy as np
from spacenet.helpers.node_node_distance import node_node_distance
from scipy.spatial import KDTree

def detour_index(spatial_network,edge_weight_name='Distance',max_distance=np.inf,low_memory=False):
    """
    Computes the detour index of a spatial network, which is defined as the average ratio of the shortest path distance between pairs of nodes to the direct distance between those nodes in space, for all pairs of nodes that are within a specified maximum distance of each other.
    The euclidean distance between nodes is used as a direct distance metric, and the shortest path distance is computed using Dijkstra's algorithm with edge weights specified by edge_weight_name.
    
    
    Parameters
    ----------
    spatial_network : networkx.Graph
        The spatial network for which to compute the detour index. The graph should have edge weights corresponding to the distance between nodes, and node attributes corresponding to the position of each node in space.
    edge_weight_name : str, optional
        The name of the edge attribute in the graph that corresponds to the distance between nodes. Default is 'Distance'.
    max_distance : float, optional
        The maximum distance to consider when computing the detour index. Only pairs of nodes that are within this distance of each other will be included in the computation. Default is np.inf (no limit).
    low_memory : bool, optional
        Whether to use a low-memory implementation of Dijkstra's algorithm that computes distances in batches. This can be useful for large networks that do not fit in memory. Default is False.
        
    Returns
    -------
    
    detour_index : float
        The detour index of the spatial network, which is defined as the average ratio of the shortest path distance between pairs of nodes to the direct distance between those nodes in space, for all pairs of nodes that are within the specified maximum distance of each other.
    """
    
    
    these_nodes = list(spatial_network.nodes())
    
    # get the network distances
    network_distances = node_node_distance(spatial_network,weight=edge_weight_name,limit=max_distance,verbose=False,low_memory=low_memory)
    
    # get the direct distance between nodes using the provided distance metric - 
    # extract the node positions from the graph
    node_positions = np.array([spatial_network.nodes[node]['position'] for node in these_nodes])
    
    # compute the distance between nodes using the provided distance metric up the maximum distance
    # using kdtree to compute the direct distances between nodes up to the maximum distance
    tree = KDTree(node_positions)
    sparse_distance_matrix = tree.sparse_distance_matrix(tree, max_distance, output_type='coo_matrix')
        
    # compute the detour index for each pair of nodes
    detour_values = []
    for i,node_i in enumerate(these_nodes):
        for j,node_j in enumerate(these_nodes):
            if i < j:
                direct_distance = sparse_distance_matrix.getrow(i).getcol(j).data
                network_distance = network_distances[node_i][node_j]
                
                if direct_distance > 0:
                    detour_values.append(network_distance / direct_distance)
    
    detour_values = np.array(detour_values) 
    detour_value = np.mean(detour_values)
    
    return detour_value
    
    
    
    
    
    