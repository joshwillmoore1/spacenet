def delete_node_labels(spatial_network,node_label_name='label'):
    """
    A function to delete node labels from a spatial network. 
    This can be used to remove categorical labels (e.g. cell types) or continuous labels (e.g. gene expression levels) from nodes in the network.
    
    Parameters
    ----------
    spatial_network : networkx.Graph
        The spatial network from which to delete node labels. Nodes should be indexed by integers.
    node_label_name : str, optional  
        The name of the node attribute corresponding to the labels to delete. Default is 'label'.
        
    Returns
    -------
    None
    
    """
    
    for node in spatial_network.nodes:
        if node_label_name in spatial_network.nodes[node]:
            del spatial_network.nodes[node][node_label_name]
            
    return None