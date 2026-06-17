import numpy as np

def query_nodes(spatial_network,node_label_name=None,relation='is',node_label_value=None):
    """
    
    Query nodes in a spatial network based on a specified node attribute, relation, and value. 
    This function allows you to filter nodes in the spatial network according to various criteria, such as equality, containment, and comparison operations.
    
    Parameters
    ----------
    
    spatial_network : networkx graph
        A spatial network represented as a NetworkX graph. Nodes should have attributes that can be queried. 
    node_label_name : str
        The name of the node attribute to query. This should be a valid node attribute in the spatial network.
    relation : str
        The relation to use for querying. Supported relations include: 'is' (or '=='), 'contains' (or 'in'), 'is not' (or '!='), 'does not contain' (or 'not in'), 'greater than' (or '>'), 'less than' (or '<'), 'greater than or equal to' (or '>='), and 'less than or equal to' (or '<=').
    node_label_value : int, float, str, or list
        The value to compare the node attribute against. The type of this value should be compatible with the type of the node attribute being queried.
    
    Returns
    -------
    np.ndarray
        An array of node indices that meet the specified relation and value criteria.
        
        
    Examples
    --------
    
    You can use the `query_nodes` function to filter nodes in a spatial network based on their attributes. Below are examples of how to use this function to query nodes with specific categorical and continuous labels.
    The first example demonstrates how to query nodes with a categorical label.
    
    .. code-block:: python  
    
        import spacenet as sn

        # get data from the Spiral dataset
        sprial_df = sn.datasets.load_dataset('spiral')
        points = sprial_df[['x','y']].values
        categorical_labels = sprial_df['Marker (categorical)'].values

        # generate a spatial network using the delaunay method and add labels
        G = sn.utils.spatial_network_from_points(points,network_type='delaunay',max_edge_distance=75)
        sn.utils.add_node_labels(G,categorical_labels,node_label_name='Marker (categorical)')

        # use query to get node ids for nodes with categorical label A
        nodes_a = sn.utils.query_nodes(G,node_label_name='Marker (categorical)',relation='is',node_label_value='A')

        print(f'First 10 Nodes with categorical label A: {nodes_a[0:10]}')
        
    
    This example shows how to query nodes with a continuous labels using relational comparisons.
    
    .. code-block:: python
    
        import spacenet as sn

        # get data from the Spiral dataset
        sprial_df = sn.datasets.load_dataset('spiral')
        points = sprial_df[['x','y']].values
        categorical_labels = sprial_df['Marker (continuous)'].values

        # generate a spatial network using the delaunay method and add labels
        G = sn.utils.spatial_network_from_points(points,network_type='delaunay',max_edge_distance=75)
        sn.utils.add_node_labels(G,categorical_labels,node_label_name='Marker (continuous)')

        # use query to get node ids for nodes with continuous label > 0.5
        nodes_of_interest = sn.utils.query_nodes(G,node_label_name='Marker (continuous)',relation='>',node_label_value=0.5)

        print(f'First 10 Nodes with continuous label > 0.5: {nodes_a[0:10]}')
    
    """
    
    # check node label name is provided and is a string and is a valid node attribute in the spatial network
    if node_label_name is None or not isinstance(node_label_name, str):
        raise ValueError("node_label_name must be provided and must be a string.")

    # check if node label name is in the available node labels in the spatial network
    available_node_labels = set()
    for node in spatial_network.nodes(data=True):
        available_node_labels.update(node[1].keys())
    if node_label_name not in available_node_labels:
        raise ValueError(f"node_label_name '{node_label_name}' is not a valid node attribute in the spatial network. Available node labels: {available_node_labels}")
    
    # check node_label_value is a number, string or list
    if not isinstance(node_label_value, (int, float, str, list)):
        raise ValueError("node_label_value must be a number, string, or list.")
    
            
    # using a node label, return a list of node indices meet the relation and value criteria
    if relation in ['is', '==']:
        node_indices = [node for node, data in spatial_network.nodes(data=True) if data.get(node_label_name) == node_label_value]
    elif relation in ['contains','in']:
        node_indices = [node for node, data in spatial_network.nodes(data=True) if node_label_value in data.get(node_label_name, '')]
    elif relation in ['is not','!=']:
        node_indices = [node for node, data in spatial_network.nodes(data=True) if data.get(node_label_name) != node_label_value]
    elif relation in ['does not contain','not in']:
        node_indices = [node for node, data in spatial_network.nodes(data=True) if node_label_value not in data.get(node_label_name, '')]
    
    # Note: The following comparisons assume that the node label values are numeric. If they are not, this will raise an error.
    elif relation in ['greater than','>']:
        node_indices = [node for node, data in spatial_network.nodes(data=True) if data.get(node_label_name) > node_label_value]
    elif relation in ['greater than or equal to','>=']:
        node_indices = [node for node, data in spatial_network.nodes(data=True) if data.get(node_label_name) >= node_label_value]
    elif relation in ['less than','<']:
        node_indices = [node for node, data in spatial_network.nodes(data=True) if data.get(node_label_name) < node_label_value]
    elif relation in ['less than or equal to','<=']:
        node_indices = [node for node, data in spatial_network.nodes(data=True) if data.get(node_label_name) <= node_label_value]
    else:
        raise ValueError(f"Unsupported relation: {relation}. The following relations are supported: 'is' (or '=='), 'contains' (or 'in'), 'is not' (or '!='), 'does not contain' (or 'not in'), 'greater than' (or '>'), 'less than' (or '<'), 'greater than or equal to' (or '>='), and 'less than or equal to' (or '<=').")
    
    return np.asarray(node_indices)
    
    