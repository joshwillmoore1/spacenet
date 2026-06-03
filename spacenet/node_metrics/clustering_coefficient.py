import networkx as nx
import numpy as np  
from spacenet.utils import add_node_labels


def clustering_coefficient(spatial_network,nodes=None,edge_weight_name='Distance',add_as_node_label=False,node_label_name='clustering'):
    """
    
    Computes the clustering coefficient for the nodes in the spatial network. 
    The clustering coefficient of a node is a measure of how connected its neighbors are to each other, and is defined as the ratio of the number of edges between the neighbors of the node to the number of possible edges between those neighbors.
    The clustering coefficient can be computed using weighted edges by specifying the name of the edge attribute to use as the weight for computing the cluster coefficient.
    
    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network for which to compute the clustering coefficient.
    nodes : list or np.ndarray, optional
        The set of nodes to compute the clustering coefficient. Default is 'None', computing for all nodes.
    edge_weight_name : str, optional
        The name of the edge attribute to use as the weight for computing the cluster coefficient. Default is 'Distance'.
    add_as_node_label : bool, optional
        Whether to add the computed clustering coefficient values as a node attribute in the spatial network. Default is False.
    node_label_name : str, optional
        The name of the node attribute to which to add the computed clustering coefficient values if add_as_node_label is True. Default is 'clustering'.
        
    Returns
    -------
    clustering_coefficients : np.ndarray
        An array of clustering coefficient values for the specified nodes in the spatial network. The order of the values corresponds to the order of the nodes in the 'nodes' parameter.
    nodes : np.ndarray
        An array of the node ids for which the clustering coefficient values were computed. The order of the nodes corresponds to the order of the values in the 'clustering_coefficients' array.
    
    
    Examples
    --------
    
    You can compute the clustering coefficient of nodes in a spatial network using the `clustering_coefficient` function. Below is an example of how to use this function to compute the clustering coefficient for all nodes in a spatial network generated from a set of points.
    
    .. code-block:: python
    
        import spacenet as sn

        # Load the spiral dataset and extract the 'x' and 'y' columns as points
        spiral_data = sn.datasets.load_dataset('spiral')
        points = spiral_data[['x', 'y']].to_numpy()

        # generate a spatial network
        G = sn.utils.generate_spatial_network(points,max_edge_distance=50)

        # compute the clustering coefficient for each node in the spatial network and add it as a node label
        cluster_vals,node_ids=sn.node_metrics.clustering_coefficient(G,add_as_node_label=True,node_label_name='clustering')

        # plot the spatial network
        sn.utils.plot_spatial_network(G,node_label_name='clustering')
    
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
    
    
    cluster_coefficients_dict = nx.clustering(spatial_network,nodes=list(nodes), weight=edge_weight_name)
    clustering_coefficients = np.array([cluster_coefficients_dict[v] for v in nodes])
    
    
    if add_as_node_label:
        add_node_labels(spatial_network,clustering_coefficients,node_label_name=node_label_name,nodes=nodes)
    
    
    return clustering_coefficients, nodes