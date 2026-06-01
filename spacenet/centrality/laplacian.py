import numpy as np
from spacenet.utils import add_node_labels
import networkx as nx


def laplacian(spatial_network,nodes=None,edge_weight_name='Distance',add_as_node_label=False,node_label_name='laplacian'):
    """
    Computes the Laplacian centrality for the nodes in the spatial network.
    The Laplacian centrality of a node is defined as the change in the Laplacian energy of the network when that node is removed.
    This measure captures the importance of a node in terms of its contribution to the overall connectivity of the network, with higher values indicating more central nodes.
    
    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network for which to calculate the Laplacian centrality.
    nodes : list or np.ndarray, optional
        The set of nodes to compute the centrality. Default is 'None', computing for all nodes.
    edge_weight_name : str, optional
        The name of the edge attribute in the graph that corresponds to the distance between nodes. Default is 'Distance'.
    add_as_node_label : bool, optional
        Whether to add the computed centrality values as a node attribute in the spatial network. Default is False.
    node_label_name : str, optional
        The name of the node attribute to which to add the computed centrality values if add_as_node_label is True. Default is 'laplacian'.
    
    
    Returns
    -------
    
    laplacian_values : np.ndarray
        An array of Laplacian centrality values for the specified nodes in the spatial network. The order of the values corresponds to the order of the nodes in the 'nodes' parameter.
    nodes : np.ndarray
        An array of the node ids for which the centrality values were computed. The order of the nodes corresponds to the order of the values in the 'laplacian_values' array.
        
    """
    # if no nodes give run on all
    if nodes is None:
        nodes = np.array(list(spatial_network.nodes))
    elif isinstance(nodes,(list,np.ndarray)):
        nodes = np.array(nodes)
        
        # validate nodes
        all_nodes = np.array(list(spatial_network.nodes))
        valid_nodes_check = np.isin(nodes,all_nodes,assume_unique=True)
        
        if np.sum(valid_nodes_check)!=len(nodes):
            raise ValueError(f'Nodes passed that are not nodes of this spatial network. These nodes are {nodes[~valid_nodes_check]}')
    else:
        raise ValueError(f'nodes not of the correct type. Acceptable types are list or array of node ids but given here f{type(nodes)}.')
    
        
    # compute the laplacian values for the network
    dict_of_lap = nx.laplacian_centrality(spatial_network, normalized=True, nodelist=list(nodes), weight=edge_weight_name)
    
    laplacian_values = np.zeros(len(nodes))
    for i,node in enumerate(nodes):
        laplacian_values[i] = dict_of_lap[node]
    
    
    if add_as_node_label:
        add_node_labels(spatial_network,laplacian_values,node_label_name=node_label_name,nodes=nodes)
        
    
    return laplacian_values, nodes