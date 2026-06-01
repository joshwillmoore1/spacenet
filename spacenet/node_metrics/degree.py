import numpy as np
from spacenet.utils import add_node_labels

def degree(spatial_network,nodes=None,edge_weight_name='Distance',add_as_node_label=False,node_label_name='degree'):
    """
    Computes the weighted degree for the nodes in the spatial network.
    
    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network for which to calculate the degree.
    nodes : list or np.ndarray, optional
        The set of nodes to compute the degree. Default is 'None', computing for all nodes.
    edge_weight_name : str, optional
        The name of the edge attribute in the graph that corresponds to the distance between nodes. Default is 'Distance'. For unweighted degree, set this to None.
    add_as_node_label : bool, optional
        Whether to add the computed degree values as a node attribute in the spatial network. Default is False.
    node_label_name : str, optional
        The name of the node attribute to which to add the computed degree values if add_as_node_label is True. Default is 'degree'.
        
    Returns
    -------
    
    degree_values : np.ndarray
        An array of degree values for the specified nodes in the spatial network. The order of the values corresponds to the order of the nodes in the 'nodes' parameter.
    nodes : np.ndarray
        An array of the node ids for which the degree values were computed. The order of the nodes corresponds to the order of the values in the 'degree_values' array.
        
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
    
    
    degree_values = np.zeros(len(nodes))
    for i, node in enumerate(nodes):
        connected_nodes = spatial_network[node]
        this_degree = 0
        for target in connected_nodes:
            if edge_weight_name in connected_nodes[target]:
                this_degree += connected_nodes[target][edge_weight_name]
            elif edge_weight_name is None:
                this_degree += 1
            else:
                raise ValueError(f'Edge weight name {edge_weight_name} not found in the edge attributes for the edge between node {node} and node {target}.')
        
        degree_values[i] = this_degree
    
    # add as node label if desired
    if add_as_node_label:
        add_node_labels(spatial_network,degree_values,node_label_name=node_label_name,nodes=nodes)
    
    
    return degree_values, nodes
    