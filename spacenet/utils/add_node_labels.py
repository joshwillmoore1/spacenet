import numpy as np
import pandas as pd


def add_node_labels(spatial_network,labels,node_label_name='label',nodes=None):
    """
    A function to add labels to nodes in a spatial network. 
    This can be used to add categorical labels (e.g. cell types) or continuous labels (e.g. gene expression levels) to nodes in the network.
    The labels will be added as a node attribute with the name specified by node_label (default is 'label').
    
    
    Parameters
    ----------
    spatial_network : networkx.Graph
        The spatial network to which to add node labels. Nodes should be indexed by integers.
    labels : array-like, np.ndarray or list
        An array-like object containing the labels to add to the nodes. The length of this should match the number of nodes to which you want to add labels.
    node_label_name : str, optional
        The name of the node attribute to which to add the labels. Default is 'label'.
    nodes : array-like, np.ndarray or list, optional
        An array-like object containing the indices of the nodes to which to add labels. If None, labels will be added to all nodes in the network. Default is None.
        
    Returns
    -------
    None
    
    """
    
    
    nodes_to_add_labels_to = []
    if nodes is None:
        nodes_to_add_labels_to = np.asarray(list(spatial_network.nodes))
    elif isinstance(nodes, np.ndarray):
        nodes_to_add_labels_to = nodes
    elif isinstance(nodes, (list,pd.arrays.StringArray, pd.Series,pd.Categorical,pd.Index)):
        nodes_to_add_labels_to = np.asarray(nodes)

    else:
        raise ValueError(f'Nodes should be a list or numpy array of node indices. Got {type(nodes)} instead.')
    
    prohibited_node_label_names = ['position','__partition_label']
    if node_label_name in prohibited_node_label_names:
        raise ValueError(f'node_label_name "{node_label_name}" is reserved and cannot be used. Prohibited node label names: {prohibited_node_label_names}')
    
    labels_to_add = []  
    if isinstance(labels, np.ndarray):
        labels_to_add = labels
    elif isinstance(labels, (list,pd.arrays.StringArray, pd.Series,pd.Categorical,pd.Index)):
        labels_to_add = np.asarray(labels)
    else:
        raise ValueError(f'Labels should be a list or numpy array of labels. Got {type(labels)} instead.')
    
    if len(nodes_to_add_labels_to) != len(labels_to_add):
        raise ValueError(f'Number of nodes to add labels to ({len(nodes_to_add_labels_to)}) should match number of labels ({len(labels_to_add)}).')
    
    for node, label in zip(nodes_to_add_labels_to, labels_to_add):
        if node not in spatial_network.nodes:
            raise ValueError(f'Node {node} is not in the spatial network.')
        spatial_network.nodes[node][node_label_name] = label
    
    return None
    
    