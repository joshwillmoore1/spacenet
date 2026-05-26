import numpy as np  


def get_nodes_and_labels(spatial_network,node_label_name='label'):
    """
    A function to extract node indices and labels from a spatial network. 
    This can be used to get the node indices and corresponding labels (e.g. cell types or gene expression levels) for nodes in the network.
    
    Parameters
    ----------
    spatial_network : networkx.Graph
        The spatial network from which to extract node indices and labels. Nodes should be indexed by integers and have a node attribute corresponding to the labels.
    node_label_name : str, optional  
        The name of the node attribute corresponding to the labels to extract. Default is 'label'.
        
    Returns
    -------
    nodes : numpy.ndarray
        An array of node indices for nodes in the spatial network that have the specified label attribute.
    labels : numpy.ndarray
        An array of labels corresponding to the specified label attribute for each node in the spatial network that has that attribute.
    
    """
    
    nodes = []
    labels = []
    
    for node in spatial_network.nodes:
        if node_label_name in spatial_network.nodes[node]:
            nodes.append(node)
            labels.append(spatial_network.nodes[node][node_label_name])
            
    return np.asarray(nodes), np.asarray(labels)