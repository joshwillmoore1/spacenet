def print_network_properties(network):
    """
    A function to print the properties of a spatial network, including the number of nodes, edges, names of edge weights, node labels names .
    
    Parameters
    ----------
    network : networkx.Graph
        The spatial network for which to print properties. Edges should have a weight attribute corresponding to the distance between nodes.
        
    Returns
    -------
    None
    
    """
    
    num_nodes = network.number_of_nodes()
    num_edges = network.number_of_edges()
    
    # Get the names of edge weights (i.e. edge attributes) if there are any edges in the network
    if num_edges > 0:
        edge_weight_names = list(list(network.edges(data=True))[0][2].keys())
    else:
        edge_weight_names = []    
    
    # get any node label names (i.e. node attributes) if there are any nodes in the network 
    node_label_names = []
    if num_nodes > 0:
        for node in network.nodes(data=True):
            node_label_names.extend(node[1].keys())
        node_label_names = list(set(node_label_names))  # get unique node label names
    
    
    print(f"------------------\nNetwork properties\n------------------\nNumber of nodes: {num_nodes}\nNumber of edges: {num_edges}\nEdge weight names: {edge_weight_names}\nNode label names: {node_label_names}\n------------------")
