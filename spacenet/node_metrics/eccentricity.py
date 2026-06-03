import numpy as np
from spacenet.utils import add_node_labels  
from spacenet.helpers import node_node_distance

def eccentricity(spatial_network,nodes=None,edge_weight_name='Distance',add_as_node_label=False,node_label_name='eccentricity'):
    """
    
    Computes the eccentricity for the nodes in the spatial network.
    The eccentricity of a node is defined as the maximum distance from that node to any other node in the network.
    It provides a measure of how far a node is from the furthest node in the network, and can be used to identify nodes that are central or peripheral in terms of their distance to other nodes.
    
    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network for which to calculate the eccentricity.
    nodes : list or np.ndarray, optional
        The set of nodes to compute the eccentricity. Default is 'None', computing for all nodes.
    edge_weight_name : str, optional
        The name of the edge attribute in the graph that corresponds to the distance between nodes. Default is 'Distance'.
    add_as_node_label : bool, optional
        Whether to add the computed eccentricity values as a node attribute in the spatial network. Default is False.
    node_label_name : str, optional
        The name of the node attribute to which to add the computed eccentricity values if add_as_node_label is True. Default is 'eccentricity'.
    
    Returns
    -------
    eccentricity_values : np.ndarray
        An array of eccentricity values for the specified nodes in the spatial network. The order of the values corresponds to the order of the nodes in the 'nodes' parameter.
    nodes : np.ndarray
        An array of the node ids for which the eccentricity values were computed. The order of the nodes corresponds to the order of the values in the 'eccentricity_values' array.
        
    Examples
    --------
    
    You can compute the eccentricity of nodes in a spatial network using the `eccentricity` function. Below is an example of how to use this function to compute the eccentricity for all nodes in a spatial network generated from a set of points.
    
    .. code-block:: python
    
        import spacenet as sn

        # Load the spiral dataset and extract the 'x' and 'y' columns as points
        spiral_data = sn.datasets.load_dataset('spiral')
        points = spiral_data[['x', 'y']].to_numpy()

        # generate a spatial network
        G = sn.utils.generate_spatial_network(points,max_edge_distance=50)

        # compute eccentricity for all nodes and add as node label
        ecc_vals,node_ids=sn.node_metrics.eccentricity(G,add_as_node_label=True,node_label_name='eccentricity')

        # plot the spatial network with the node label 'eccentricity'
        sn.utils.plot_spatial_network(G,node_label_name='eccentricity')
    
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
    
    
    network_distances = node_node_distance(spatial_network,sources=nodes,weight=edge_weight_name,limit=np.inf)
    
    eccentricity_values = np.zeros(len(nodes))
    for i, node in enumerate(nodes):
        these_distances = network_distances[node]
        this_eccentricity = max(these_distances.values())
        eccentricity_values[i] = this_eccentricity

    if add_as_node_label:
        add_node_labels(spatial_network,eccentricity_values,node_label_name=node_label_name,nodes=nodes)
    return eccentricity_values, nodes