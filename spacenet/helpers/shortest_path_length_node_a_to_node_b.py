import networkx as nx

def shortest_path_length_node_a_to_node_b(spatial_network,node_a,node_b,edge_weight_name='Distance'):
    """
    
    Computes the shortest path distance from node_a to node_b in a spatial network using Dijkstra's algorithm, with the specified edge weight. 
    The function first checks if the distance from node_a to node_b is already cached in the spatial network's distance cache for the given edge weight. 
    If it is cached, it returns the cached distance. If not, it computes the shortest path distance using Dijkstra's algorithm and returns the result.
    
    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network for which to compute the shortest path distance.
    node_a : int or other
        The index of the source node. This can be an integer or any other hashable type that represents a node in the spatial network.
    node_b : int or other
        The index of the target node. This can be an integer or any other hashable type that represents a node in the spatial network.
    edge_weight_name : str, optional
        The name of the edge attribute to use as the weight for computing the shortest path distance. Default is 'Distance'.
    
    Returns
    -------
    shortest_path_distance : float
        The shortest path distance from node_a to node_b in the spatial network, computed using Dijkstra's algorithm with the specified edge weight.
    
    
    """
    
    # first check the cache for this weight
    if edge_weight_name in spatial_network.distance_cache:
        cached_distances = spatial_network.distance_cache[edge_weight_name]['distances']
        
        # if the cache exists, check if the distance from node_a to node_b is in the cache
        if node_a in cached_distances and node_b in cached_distances[node_a]:
            return cached_distances[node_a][node_b]
        if node_b in cached_distances and node_a in cached_distances[node_b]:
            return cached_distances[node_b][node_a]
        
    # if not in cache, compute the shortest path distance using Dijkstra's algorithm
    node_node_shorest_path=nx.shortest_path_length(spatial_network, source=node_a, target=node_b, weight=edge_weight_name, method='dijkstra')
        
    return node_node_shorest_path