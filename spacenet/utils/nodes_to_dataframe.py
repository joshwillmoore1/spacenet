import pandas as pd


def nodes_to_dataframe(spatial_network):
    """
    Converts the nodes of a spatial network into a pandas DataFrame.
    The resulting DataFrame is indexed by node ids and has columns for each attribute of the nodes in the graph.
    If an attribute is missing for a node, the corresponding cell will be filled with NaN.
    
    Parameters
    ----------
    spatial_network : NetworkX graph
        The spatial network for which to create the node dataframe.
        
    Returns
    -------
    node_df : pandas DataFrame
        A dataframe indexed by node ids with columns for each attribute of the nodes in the graph. If an attribute is missing for a node, the corresponding cell will be filled with NaN.
    """
    
    
    
    # make a dataframe indexed by nodes with columns for each attribute of the nodes in the graph
    # if an attribute is missing for a node, fill with NaN
    
    node_attributes = {}
    for node in spatial_network.nodes(data=True):
        node_id = node[0]
        attributes = node[1]
        node_attributes[node_id] = attributes
    node_df = pd.DataFrame.from_dict(node_attributes, orient='index')
    
    # name the index with the node id
    node_df.index.name = 'node_id'
    return node_df