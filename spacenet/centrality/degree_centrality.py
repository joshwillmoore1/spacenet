from spacenet.node_metrics.degree import degree
from spacenet.global_metrics import volume
from spacenet.utils import add_node_labels

def degree_centrality(spatial_network,nodes=None,edge_weight_name='Distance',add_as_node_label=False,node_label_name='degree'):
    """
    Computes the degree centrality for the nodes in the spatial network.
    The degree centrality of a node is defined as the sum of the weights of the edges connected to that node, normalized by the total volume of the network.
    
    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network for which to calculate the degree centrality.
    nodes : list or np.ndarray, optional
        The set of nodes to compute the centrality. Default is 'None', computing for all nodes.
    edge_weight_name : str, optional
        The name of the edge attribute in the graph that corresponds to the distance between nodes. Default is 'Distance'.
    add_as_node_label : bool, optional
        Whether to add the computed centrality values as a node attribute in the spatial network. Default is False.
    node_label_name : str, optional
        The name of the node attribute to which to add the computed centrality values if add_as_node_label is True. Default is 'degree'.
    
    
    Returns
    -------
    
    degree_values : np.ndarray
        An array of degree centrality values for the specified nodes in the spatial network. The order of the values corresponds to the order of the nodes in the 'nodes' parameter.
    nodes : np.ndarray
        An array of the node ids for which the centrality values were computed. The order of the nodes corresponds to the order of the values in the 'degree_values' array.
        
    """
    
    # compute the degree values for the nodes
    degree_values , nodes = degree(spatial_network,nodes=nodes,edge_weight_name=edge_weight_name,add_as_node_label=False)
    
    # normalise for the total possible degree
    total_volume = volume(spatial_network,edge_weight_name=edge_weight_name)
    degree_values = degree_values / total_volume
    
    if add_as_node_label:
        add_node_labels(spatial_network,degree_values,node_label_name=node_label_name,nodes=nodes)
    
    
    return degree_values, nodes